from sys import argv
import os
from os import chdir, path
import numpy as np
from oxDNA_analysis_tools import duplex_finder, output_bonds
from oxDNA_analysis_tools.UTILS.RyeReader import strand_describe, describe, get_confs, inbox
from copy import deepcopy
import json as js
import sys
from contextlib import contextmanager
from scipy import stats

def oxDNA_to_nNxB(particles_per_course_bead, path_to_conf, path_to_top, path_to_input, path_to_traj, material, remainder_modifer, n_cpus=1, system_name=None):
    """
    Converts oxDNA simulation output to nNxB format for coarse-grained analysis.

    Parameters:
    - particles_per_course_bead (int): Number of particles per coarse-grained bead.
    - path_to_conf (str): Path to the configuration file.
    - path_to_top (str): Path to the topology file.
    - path_to_input (str): Path to the input file for oxDNA.
    - path_to_traj (str): Path to the trajectory file from oxDNA simulation.
    - n_cpus (int): Number of CPUs to use for parallel processing (default: 1).
    - system_name (str): Name of the system for output files (default: None).

    Returns:
    - None, but prints the status of nNxB file creation and any issues encountered.
    """
    monomer_id_info, positions = nucleotide_and_position_info(path_to_conf, path_to_top)
    path_to_duplex_file, path_to_hb_file = run_oat_duplex_output_bonds(path_to_input, path_to_traj, n_cpus)
    print('Course-graining the particles')
    duplex_to_particle, particle_to_duplex = associate_particle_idx_to_unique_duiplex(path_to_duplex_file)
    duplex_to_particle, all_edge_cases = run_duplex_finder_error_correction(duplex_to_particle, particle_to_duplex, monomer_id_info, positions, path_to_hb_file)

    if system_name is None:
        system_name = path_to_conf.split('/')[-1].split('.')[0]
        
    coarse_particles_positions, coarse_particles_nucleotides, coarse_particle_indexes, course_particle_strands = create_coarse_particle_info(
       duplex_to_particle, positions, particles_per_course_bead, monomer_id_info, remainder_modifer=remainder_modifer 
    )
    coarse_particles_nucleotides_ordered, coarse_particles_positions_ordered, bead_pair_dict, coarse_particle_indexes_ordered, formatted_strand_list = write_course_particle_files_functional(
       coarse_particles_nucleotides, coarse_particles_positions, coarse_particle_indexes, course_particle_strands, system_name, particles_per_course_bead, material
    )
    if all_edge_cases:
        return print('Able to create nNxB files, but unable to assign all nucleotides to duplexes.')
    else:
        return print('nNxB files created successfully.')

    
def nucleotide_and_position_info(path_to_conf, path_to_top):
    """
    Extracts nucleotide and position information from configuration and topology files.

    Parameters:
    - path_to_conf: Path to the configuration file.
    - path_to_top: Path to the topology file.

    Returns:
    - Tuple containing monomer ID information and positions.
    """
    top_info, traj_info = describe(None, path_to_conf)
    system, monomer_id_info = strand_describe(path_to_top)
    
    ox_conf = get_confs(top_info, traj_info, 0, 1)[0]
    ox_conf = inbox(ox_conf, center=True)
    positions = ox_conf.positions
    
    return monomer_id_info, positions


def run_oat_duplex_output_bonds(path_to_input, path_to_traj, n_cpus):
    """
    Runs the duplex finder and output bonds analysis using oxDNA Analysis Tools.

    Parameters:
    - path_to_input: Path to the input file for oxDNA.
    - path_to_traj: Path to the trajectory file from oxDNA simulation.
    - n_cpus: Number of CPUs to use for parallel processing.

    Returns:
    - Paths to the duplex info file and hydrogen bonds info file.
    """
    path_to_files = '/'.join(path_to_traj.split('/')[:-1])
    path_to_duplex_file = create_duplex_info_file(path_to_files, path_to_input, path_to_traj, n_cpus=n_cpus)
    path_to_hb_file = create_output_bonds_info_file(path_to_files, path_to_input, path_to_traj, n_cpus=n_cpus)
    
    return path_to_duplex_file, path_to_hb_file 


def create_duplex_info_file(abs_path_to_files, input_file_name, conf_file_name, n_cpus=1):
    print('Creating duplex info file')
    chdir(abs_path_to_files)
    argv.clear()
    argv.extend(['duplex_finder.py', input_file_name, conf_file_name, '-p', str(n_cpus)])
    duplex_finder.main()
    path_to_duplex_info_file = path.join(abs_path_to_files, 'angles.txt')
    return path_to_duplex_info_file


def create_output_bonds_info_file(abs_path_to_files, input_file_name, conf_file_name, n_cpus=1):
    print('Creating hydrogen bonds info file')
    chdir(abs_path_to_files)
    argv.clear()
    argv.extend(['output_bonds.py', '-v', 'bonds.json',input_file_name, conf_file_name, '-p', str(n_cpus)])
    output_bonds.main()
    path_to_hb_info_file = path.join(abs_path_to_files, 'bonds_HB.json')
    return path_to_hb_info_file

def read_hb_energy_file(path_to_hb_energy_file):
    with open(path_to_hb_energy_file, 'r') as f:
        hb_energy = js.load(f)
    return hb_energy

def associate_particle_idx_to_unique_duiplex(path_to_duplex_info:str) -> dict:
    """
    Returns: Dictonary with duplex id as key, and as values we have a list of 2 lists where the first
    list is the particle idxes of the duplex in 3` -> 5` and 2nd list is particle idex in 5` -> 3`
    """

    with open(path_to_duplex_info, 'r') as f:
        duplex_ends = f.readlines()

    n_conf = duplex_ends[-1].split('\t')[0]
    d1_ends = {}
    d2_ends = {}
    for conf in range(0, int(n_conf)+1):
        d1_ends[conf] = [list(map(int, d.split('\t')[2:4])) for d in duplex_ends[1:] if int(d.split('\t')[0]) == conf]
        d2_ends[conf] = [list(map(int, d.split('\t')[4:6])) for d in duplex_ends[1:] if int(d.split('\t')[0]) == conf]

    d1s_range = {k:[list(range(d[0], d[1]+1)) for d in val] for k, val in d1_ends.items()}
    d2s_range = {k:[list(range(d[0], d[1]+1)) for d in val] for k, val in d2_ends.items()}

    d1_most_nucs = {k:len(np.concatenate(vals)) for k, vals in d1s_range.items()}
    d2_most_nucs = {k:len(np.concatenate(vals)) for k, vals in d2s_range.items()}
    
    d1_m = list(d1_most_nucs.values())
    d2_m = list(d2_most_nucs.values())

    d1_argmax = np.argmax(d1_m)
    d2_argmax = np.argmax(d2_m)
    d1_max = np.max(d1_m)
    d2_max = np.max(d2_m)
    if d1_argmax == d2_argmax:
        d1s = d1s_range[d1_argmax]
        d2s = d2s_range[d2_argmax]
    else:
        d1s = d1s_range[d1_argmax]
        d2s = d2s_range[d1_argmax]

    # else:
    #     d1s = d1s_range[d2_argmax]
    #     d2s = d2s_range[d2_argmax]

    p_to_d = {}
    for i, (d1, d2) in enumerate(zip(d1s, d2s)):
        for p in d1:
            p_to_d[p] = i+1
        for p in d2:
            p_to_d[p] = -(i+1)

    d_to_p = {duplex: [[],[]] for duplex in range(1, len(d1s)+1,)}

    # d_to_p
    for particle, duplex in p_to_d.items():
        if duplex > 0:
            d_to_p[duplex][0].append(particle)
        elif duplex < 0:
            d_to_p[abs(duplex)][1].append(particle)

    # Now d_to_p will contain lists of particles for each duplex
    for duplex, strands in d_to_p.items():
        strands[1] = strands[1][::-1]
        
    return d_to_p, p_to_d


def run_duplex_finder_error_correction(duplex_to_particle, particle_to_duplex, nucleotides_in_duplex, positions, path_to_hb_file):
    """
    Runs error correction for the duplex finder by handling edge cases and non-bonded fixes.

    Parameters:
    - duplex_to_particle (dict): Mapping from duplexes to particles.
    - particle_to_duplex (dict): Mapping from particles to duplexes.
    - nucleotides_in_duplex: Information about nucleotides in each duplex.
    - positions (np.array): Positions of particles.
    - path_to_hb_file (str): Path to the hydrogen bond information file.

    Returns:
    - Updated duplex_to_particle mapping and a list of all edge cases.
    """
    all_edge_cases, len_one_parts = get_nuc_not_included_in_d_to_p(particle_to_duplex, nucleotides_in_duplex)

    if len_one_parts.size > 0:
        duplex_to_particle, single_nucs_dealt_with = deal_with_single_nuc_edge_cases(positions, len_one_parts, duplex_to_particle)

        if type(single_nucs_dealt_with) != bool:
            for idx, nucs in enumerate(all_edge_cases):
                for i, val in enumerate(nucs):
                    if val in single_nucs_dealt_with:
                        all_edge_cases[idx].pop(i)

            all_edge_cases = [sublist for sublist in all_edge_cases if sublist]
    
    if all_edge_cases:
        duplex_to_particle, fixed = fully_complementary_sequential_fix(nucleotides_in_duplex, positions, all_edge_cases, duplex_to_particle)
        if fixed:
            values_to_remove = sum(fixed, [])
            all_edge_cases = [[val for val in sublist if val not in values_to_remove] for sublist in all_edge_cases]
            all_edge_cases = [sublist for sublist in all_edge_cases if sublist]   
        
    # duplex_to_particle, fixed = fully_complementary_sequential_fixs(nucleotides_in_duplex, positions, all_edge_cases, duplex_to_particle, path_to_hb_info_file)
    
    if all_edge_cases:
        duplex_to_particle, fixed = non_bonded_fixes(path_to_hb_file, all_edge_cases, duplex_to_particle)
        if fixed:
            values_to_remove = np.concatenate(fixed)
            all_edge_cases = [[val for val in sublist if val not in values_to_remove] for sublist in all_edge_cases]
            all_edge_cases = [sublist for sublist in all_edge_cases if sublist]

    if all_edge_cases:
        print('Unable to assign all nucleotides to duplexes. Continuing with the ones that were assigned. Nucleotides not assigned to duplexes are:\n', all_edge_cases)
    
    return duplex_to_particle, all_edge_cases


def get_nuc_not_included_in_d_to_p(p_to_d, nucleotides_in_duplex):
    
    nucs_in_duplex = list(p_to_d.keys())
    nucs_in_duplex.sort()
    ids = [nuc.id for nuc in nucleotides_in_duplex]
    ids.sort()
    ids = set(ids)
    nucs_in_duplex = set(nucs_in_duplex)
    difference = ids.difference(nucs_in_duplex)
    difference_list = sorted(list(difference))

    sequential_parts = []
    current_sequence = [difference_list[0]]

    for i in range(1, len(difference_list)):
        if difference_list[i] == difference_list[i - 1] + 1:
            # The current element is sequential to the previous one
            current_sequence.append(difference_list[i])
        else:
            # The current element is not sequential to the previous one,
            # store the current sequence and start a new one
            sequential_parts.append(current_sequence)
            current_sequence = [difference_list[i]]

    # Append the last sequence if it exists
    if current_sequence:
        sequential_parts.append(current_sequence)

    len_parts = [len(part) for part in sequential_parts]

    len_one_parts = np.array([part for part in sequential_parts if len(part) == 1]).flatten()
    
    return sequential_parts, len_one_parts


def calculate_distance_matrix(points):
    # Ensure the input is a NumPy array
    points = np.asarray(points)

    # Calculate differences in each dimension (x, y, z) between points
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]

    # Calculate the squared distances
    squared_distances = np.sum(diff**2, axis=-1)

    # Take the square root to get the Euclidean distance
    distance_matrix = np.sqrt(squared_distances)

    return distance_matrix


def deal_with_single_nuc_edge_cases(positions, len_one_parts, d_to_p):
    
    len_one_parts_pos = positions[len_one_parts]

    len_one_parts_pos_ma = calculate_distance_matrix(len_one_parts_pos)
    len_one_parts_pos_ma[len_one_parts_pos_ma == 0] = 'inf'
    idx = np.argmin(len_one_parts_pos_ma, axis=1)
    mins = np.min(len_one_parts_pos_ma, axis=1)

    idx_pairs = [[i,val] for i,val in enumerate(idx)]

    idx_pair_set = []
    for pair in idx_pairs:
        pair_inv = [pair[1], pair[0]]
        if pair_inv not in idx_pair_set:
            idx_pair_set.append(pair)

    len_one_parts_pairs = [[len_one_parts[pair[0]], len_one_parts[pair[1]]] for pair in idx_pair_set]
    removed = []

    for key, value in d_to_p.items():
        ends = [value[0][-1], value[1][-1]]
        starts = [value[0][0], value[1][0]]
        for pairs in  len_one_parts_pairs:
            look_0 = [pairs[0] - 1, pairs[1] + 1]
            look_1 = [pairs[0] + 1, pairs[1] - 1]

            if ends == look_0:
                d_to_p[key][0].append(pairs[0])
                d_to_p[key][1].append(pairs[1])
                removed.append(pairs)

            elif starts == look_0:
                d_to_p[key][0].insert(0, pairs[0])
                d_to_p[key][1].insert(0, pairs[1])
                removed.append(pairs)

            elif ends == look_1:
                d_to_p[key][0].append(pairs[0])
                d_to_p[key][1].append(pairs[1])
                removed.append(pairs)

            elif starts == look_1:
                d_to_p[key][0].insert(0, pairs[0])
                d_to_p[key][1].insert(0, pairs[1])
                removed.append(pairs)

    if removed:
        single_nucs_dealt_with = np.concatenate(removed)
    else:
        single_nucs_dealt_with = False
    
    return d_to_p, single_nucs_dealt_with


def full_sequential_group_distance_check(positions, sequential_parts):
    pos_edge_cases = [positions[part] for part in sequential_parts]
    cms_edge_cases = np.array([np.mean(part, axis=0) for part in pos_edge_cases])

    distance_matrix = calculate_distance_matrix(cms_edge_cases)
    distance_matrix[distance_matrix == 0] = 'inf'

    nearest_edge_case = np.argmin(distance_matrix, axis=1)
    nearest_edge_case_min = np.min(distance_matrix, axis=1)

    nearest_edge_case_pairs = [[idx,pair] for idx,pair in enumerate(nearest_edge_case)]

    pair_set = []
    for pair in nearest_edge_case_pairs:
        pair_inv = [pair[1], pair[0]]
        if pair_inv not in pair_set:
            pair_set.append(pair)

    indexes = [[sequential_parts[pair[0]], sequential_parts[pair[1]]] for pair in pair_set]

    return pair_set, indexes, nearest_edge_case_min


def fully_complementary_sequential_fix(nucleotides_in_duplex, positions, all_edge_cases, duplex_to_particle):
    pair_set, indexes, nearest_edge_case_min = full_sequential_group_distance_check(positions, all_edge_cases)

    easy_fix = []
    monomer_info = np.array(nucleotides_in_duplex)
    for idxes in indexes:
        nuc_dict = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
        idx_1 = idxes[0]
        idx_2 = idxes[1]

        monomers_1 = monomer_info[idx_1]
        monomers_1 = [mono.btype for mono in monomers_1]

        monomers_2 = monomer_info[idx_2]
        monomers_2 = [nuc_dict[mono.btype] for mono in monomers_2][::-1]

        if monomers_1 == monomers_2:
            new_key = len(duplex_to_particle) + 1
            duplex_to_particle[int(new_key)] = [idxes[0], idxes[1][::-1]]
            easy_fix.append([idxes[0], idxes[1][::-1]])
    
    return duplex_to_particle, easy_fix


def non_bonded_fixes(path_to_hb_info_file, all_edge_cases, duplex_to_particle):
    fixed = []
    hb_energy = read_hb_energy_file(path_to_hb_info_file)
    hb_energy = np.array(hb_energy['HB (oxDNA su)'])
    hb_boolean = np.array(hb_energy < -0.2)

    hb_boolean_edge_cases = [hb_boolean[case] for case in all_edge_cases]
    new_boolean_criteria = [np.mean(~boolean) for boolean in hb_boolean_edge_cases]
    hb_edge_cases_idx = [edge_case for edge_case, boolean in zip(all_edge_cases, new_boolean_criteria) if boolean >= 0.3]
    
    for case in hb_edge_cases_idx:
            new_key = len(duplex_to_particle) + 1
            duplex_to_particle[int(new_key)] = [case]
            
            fixed.append(case)
    
        
    return duplex_to_particle, fixed


def fully_complementary_sequential_fixs(nucleotides_in_duplex, positions, all_edge_cases, duplex_to_particle, path_to_hb_info_file):
    pair_set, indexes, nearest_edge_case_min = full_sequential_group_distance_check(positions, all_edge_cases)
    nuc_dict = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    easy_fix = []
    alignments1 = []
    alignments2 = []
    monomer_info = np.array(nucleotides_in_duplex)
    
    hb_energy = read_hb_energy_file(path_to_hb_info_file)
    hb_energy = np.array(hb_energy['HB (oxDNA su)'])
    for idxes in indexes:
        idx_1 = idxes[0]
        idx_2 = idxes[1]

        monomers_1 = monomer_info[idx_1]
        monomers_1 = ''.join([mono.btype for mono in monomers_1])

        monomers_2 = monomer_info[idx_2]
        monomers_2 = ''.join([nuc_dict[mono.btype] for mono in monomers_2][::-1])
        
        alignment1, alignment2, seq_1_indexs, seq_2_indexs = smith_waterman(monomers_1, monomers_2)
        
        # if len(seq_1_indexs) > 1:
        # print(seq_1_indexs, seq_2_indexs)
        # print(idx_1, idx_2)
        print(alignment1)
        print(monomers_1, monomers_2)
        
        aligned_idxes_1 = [idx_1[i] for i in seq_1_indexs]
        aligned_idxes_2 = [idx_2[i] for i in seq_2_indexs]
        
        energies_1 = hb_energy[aligned_idxes_1]
        energies_2 = hb_energy[aligned_idxes_2]
        print(aligned_idxes_1, aligned_idxes_2)
        print(energies_1, energies_2)
        
        # new_key = len(duplex_to_particle) + 1
        # duplex_to_particle[int(new_key)] = [aligned_idxes_1, aligned_idxes_2[::-1]]
        # easy_fix.append([aligned_idxes_1, aligned_idxes_2[::-1]])


    return duplex_to_particle, easy_fix



def smith_waterman(seq1, seq2, match_score=1, mismatch_penalty=-1, gap_penalty=-2):
    # Initialize the matrix
    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m+1, n+1))

    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                diag_score = score_matrix[i-1][j-1] + match_score
            else:
                diag_score = score_matrix[i-1][j-1] + mismatch_penalty

            score_matrix[i][j] = max(
                0,
                diag_score,
                score_matrix[i-1][j] + gap_penalty,
                score_matrix[i][j-1] + gap_penalty
            )

    # Trace back from the highest scoring cell
    alignment1, alignment2 = '', ''
    i, j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)
    seq_1_indexs = []
    seq_2_indexs = []

    while score_matrix[i][j] > 0:
        if i > 0 and score_matrix[i][j] == score_matrix[i-1][j] + gap_penalty:
            alignment1 = seq1[i-1] + alignment1
            alignment2 = '-' + alignment2
            seq_1_indexs.append(i-1)
            i -= 1
        elif j > 0 and score_matrix[i][j] == score_matrix[i][j-1] + gap_penalty:
            alignment1 = '-' + alignment1
            alignment2 = seq2[j-1] + alignment2
            seq_2_indexs.append(j-1)
            j -= 1
        else:
            alignment1 = seq1[i-1] + alignment1
            alignment2 = seq2[j-1] + alignment2
            seq_1_indexs.append(i-1)
            seq_2_indexs.append(j-1)
            i -= 1
            j -= 1
    seq_1_indexs = seq_1_indexs[::-1]
    seq_2_indexs = seq_2_indexs[::-1]
    
    return alignment1, alignment2, seq_1_indexs, seq_2_indexs


def create_coarse_particle_info(d_to_p, positions, particles_per_course_bead, nucleotides_in_duplex, remainder_modifer=0.25):
    remainder_modifer = np.floor(particles_per_course_bead * remainder_modifer)
    coarse_particles_positions = {}
    coarse_particles_nucleotides = {}
    coarse_particle_indexes = {}
    course_particle_strands = {}
    
    # Iterate through each duplex
    # duplex is the int starting at 1, strand is each strand in the duplex
    for duplex, strands in d_to_p.items():
        # strands[0] contains the particle indices of the first strand
        # strands[1] contains the particle indices of the second strand
        
        for strand_idx, strand in enumerate(strands):
            #start by pulling the first strand from strands
            coarse_positions = []
            coarse_nucleotides = []
            course_indexes = []
            course_strands = []
            
            # Iterate through the particles in the strand, taking N at a time
            for i in range(0, len(strand), particles_per_course_bead):
                # Get the particle indices of the current group
                particle_indices = strand[i:i+particles_per_course_bead]

                if (len(particle_indices) > remainder_modifer) or (not course_indexes):
                    course_indexes.append(particle_indices)
                    # Compute the center of mass for the current group
                    center_of_mass = np.mean(positions[particle_indices], axis=0)
                    coarse_positions.append(center_of_mass)

                    # Accumulate the nucleotide types for the current group
                    nucleotide_types = ''.join([nucleotides_in_duplex[idx].btype for idx in particle_indices])
                    coarse_nucleotides.append(nucleotide_types)
                    
                    strand_ids = stats.mode([nucleotides_in_duplex[idx].strand.id for idx in particle_indices], keepdims=False)[0]
                    course_strands.append(strand_ids)
                    
                elif len(particle_indices) <= remainder_modifer:
                    course_indexes[-1].extend(particle_indices)

                    # Accumulate the nucleotide types for the current group
                    nucleotide_types = ''.join([nucleotides_in_duplex[idx].btype for idx in particle_indices])
                    coarse_nucleotides[-1] += nucleotide_types
    
                    # Compute the center of mass for the current group
                    particle_indices = strand[i-particles_per_course_bead:]
                    center_of_mass = np.mean(positions[particle_indices], axis=0)
                    coarse_positions[-1] = center_of_mass
                    
                    strand_ids = stats.mode([nucleotides_in_duplex[idx].strand.id for idx in particle_indices], keepdims=False)[0]
                    course_strands[-1] = (strand_ids)
                else:
                    print(f'Error: Something went wrong with the coarse-graining process. Duplex idx: {duplex}')
                    
                    
            # Store the coarse-grained positions for the current strand in the duplex
            key = (duplex, strand_idx)
            coarse_particles_positions[key] = coarse_positions
            coarse_particles_nucleotides[key] = coarse_nucleotides
            coarse_particle_indexes[key] = course_indexes
            course_particle_strands[key] = course_strands
    
    return coarse_particles_positions, coarse_particles_nucleotides, coarse_particle_indexes, course_particle_strands


def write_course_particle_files_functional(coarse_particles_nucleotides, coarse_particles_positions, coarse_particle_indexes, course_particle_strands, system_name, particles_per_course_bead, material):
    write_sanity_check(coarse_particles_nucleotides, system_name)
    
    keys_ordered_acending, indexes_ordered_acending = order_indexes_and_keys_acending(coarse_particle_indexes)
    
    coarse_particles_nucleotides_ordered = write_course_particle_nucleotides(coarse_particles_nucleotides, coarse_particle_indexes, keys_ordered_acending, system_name, particles_per_course_bead)
    coarse_particles_positions_ordered = write_course_particles_positions(coarse_particles_positions, keys_ordered_acending, system_name, particles_per_course_bead)
    bead_pair_dict, coarse_particle_indexes_ordered = write_course_particle_bonded_pairs(coarse_particle_indexes, coarse_particles_nucleotides_ordered, indexes_ordered_acending, system_name, particles_per_course_bead)
    strand_list = write_strand_info(course_particle_strands, coarse_particles_nucleotides, keys_ordered_acending, system_name, particles_per_course_bead, material)
    
    return coarse_particles_nucleotides_ordered, coarse_particles_positions_ordered, bead_pair_dict, coarse_particle_indexes_ordered, strand_list


def order_indexes_and_keys_acending(coarse_particle_indexes):
    indexes_ordered_acending = []
    for values in coarse_particle_indexes.values():
        indexes_ordered_acending.append(sorted(values))
    indexes_ordered_acending.sort()
    
    keys_ordered_acending = []
    for lists in indexes_ordered_acending:
        for keys, values in coarse_particle_indexes.items():
            values = sorted(values)
            if lists == values:
                keys_ordered_acending.append(keys)
                
    return keys_ordered_acending, indexes_ordered_acending


def write_strand_info(course_particle_strands, coarse_particles_nucleotides, keys_ordered_acending, system_name, particles_per_course_bead, material):  
    ordered_nucs = deepcopy(coarse_particles_nucleotides)
    
    # Reverse the nucleotides in each group to put the nucleotides back
    # in 3` to 5` order after I reversed them in d_to_p
    course_keys = list(coarse_particles_nucleotides.keys())
    num_course_duplex = len([key for key in course_keys if key[1] == 1])
    
    for idx in range(num_course_duplex):
        for lists in range(len(coarse_particles_nucleotides[(idx+1,1)])):
            ordered_nucs[(idx+1,1)][lists] = coarse_particles_nucleotides[(idx+1,1)][lists][::-1]

    # Reverse the order of the groups to put them back in 3` to 5` order
    for idx in range(num_course_duplex):
        ordered_nucs[(idx+1,1)] = coarse_particles_nucleotides[(idx+1,1)][::-1]
    
    sort_particles_based_on_strand = {}
    
    for key in keys_ordered_acending:
        strands = course_particle_strands[key]
        nucs = ordered_nucs[key]
        
        for strand, nuc in zip(strands, nucs):
            if strand not in sort_particles_based_on_strand:
                sort_particles_based_on_strand[strand] = []
            sort_particles_based_on_strand[strand].append(nuc)
    
    sorted_keys = sorted(sort_particles_based_on_strand.keys())
    formatted_list = []
    for key in sorted_keys:
        list_of_nucs = sort_particles_based_on_strand[key]
        formatted_list.extend(['x',])
        formatted_list.extend(list_of_nucs)
        formatted_list.extend(['x',])
    
    five_to_three_formatted_list = []
    for nucs in formatted_list:
        five_to_three_formatted_list.append(nucs[::-1])
    five_to_three_formatted_list = five_to_three_formatted_list[::-1]

    write_annamo_topology(five_to_three_formatted_list, system_name, particles_per_course_bead)
    write_annamo_interaction_matrix(five_to_three_formatted_list, material)

    
    with open(f'{system_name}_{particles_per_course_bead}_nuc_beads_strands_list.txt', 'w') as f:
        f.write(f'{five_to_three_formatted_list}')
    
    return formatted_list

def write_course_particle_nucleotides(coarse_particles_nucleotides, coarse_particle_indexes, keys_ordered_acending, system_name, particles_per_course_bead):
    ordered_nucs = deepcopy(coarse_particles_nucleotides)
    
    # Reverse the nucleotides in each group to put the nucleotides back
    # in 3` to 5` order after I reversed them in d_to_p
    course_keys = list(coarse_particles_nucleotides.keys())
    num_course_duplex = len([key for key in course_keys if key[1] == 1])
    
    for idx in range(num_course_duplex):
        for lists in range(len(coarse_particles_nucleotides[(idx+1,1)])):
            ordered_nucs[(idx+1,1)][lists] = coarse_particles_nucleotides[(idx+1,1)][lists][::-1]

    # Reverse the order of the groups to put them back in 3` to 5` order
    for idx in range(num_course_duplex):
        ordered_nucs[(idx+1,1)] = coarse_particles_nucleotides[(idx+1,1)][::-1]

    # Order the groups by acending particle index
    coarse_particles_nucleotides_ordered = [ordered_nucs[order] for order in keys_ordered_acending]   
    
    nuc_beads_dic = []
    
    for lists in coarse_particles_nucleotides_ordered:
        for string in lists:
            nuc_beads_dic.append(string)
    nuc_beads_dic = {f'{key}':string for key, string in enumerate(nuc_beads_dic)}

    with open(f'{system_name}_{particles_per_course_bead}_nuc_beads_sequence.txt', 'w') as f:
        for key, value in nuc_beads_dic.items():
            f.write(f'{key} {value}\n')

    return coarse_particles_nucleotides_ordered

def write_course_particles_positions(coarse_particles_positions, keys_ordered_acending, system_name, particles_per_course_bead):
    ordered_positions = deepcopy(coarse_particles_positions)    
    
    course_keys = list(coarse_particles_positions.keys())
    num_course_duplex = len([key for key in course_keys if key[1] == 1])
    # Reverse the order of the groups to put them back in 3` to 5` order
    for idx in range(num_course_duplex):
        ordered_positions[(idx+1,1)] = coarse_particles_positions[(idx+1,1)][::-1]

    coarse_particles_positions_ordered = [ordered_positions[order] for order in keys_ordered_acending]    
    
    with open(f'{system_name}_{particles_per_course_bead}_nuc_beads_positions.xyz', 'w') as file:
        for bead_positions in coarse_particles_positions_ordered:
            for position in bead_positions:
                # Write x, y, z positions separated by spaces
                file.write(f"{position[0]} {position[1]} {position[2]}\n")
    
    return coarse_particles_positions_ordered


def write_course_particle_bonded_pairs(coarse_particle_indexes, coarse_particles_nucleotides_ordered, indexes_ordered_acending, system_name, particles_per_course_bead):
    ordered_indexes = deepcopy(coarse_particle_indexes)
    
    course_keys = list(coarse_particle_indexes.keys())
    num_course_duplex = len([key for key in course_keys if key[1] == 1])

    ordered_indexes_values = list(ordered_indexes.values())
    particles_idx_pair_dict = {}
    i = 0
    for values in ordered_indexes_values[:num_course_duplex*2:2]:
        for bead in values:
            particles_idx_pair_dict[i] = [bead]
            i += 1
            
    i = 0
    for values in ordered_indexes_values[1:num_course_duplex*2:2]:
        for bead in values:
            particles_idx_pair_dict[i].append(bead)
            i += 1
    
    start = len(particles_idx_pair_dict)
    for values in ordered_indexes_values[num_course_duplex*2:]:
        for bead in values:
            particles_idx_pair_dict[start] = [bead]
            start += 1
    
    # print(particles_idx_pair_dict)
    
    indexes_acending = []
    for lists in indexes_ordered_acending:
        for embedded_list in lists:
            indexes_acending.append(embedded_list)
    
    
    ordered_indexes = {f'{key}':ind for key,ind in enumerate(indexes_acending)}
    
    dict_keys = list(ordered_indexes.keys())
    dict_values = list(ordered_indexes.values())
    
    # print(ordered_indexes)
    
    course_bead_idx_pair_dict = {}
    i = 0
    for beads in particles_idx_pair_dict.values():
        try:
            course_bead_idx_pair_dict[i] = (dict_values.index(beads[0]), dict_values.index(beads[1]))
        except:
            course_bead_idx_pair_dict[i] = (dict_values.index(beads[0]),)
        i +=1
    
    nucs = [item for sublist in coarse_particles_nucleotides_ordered for item in sublist]
    
    with open(f'{system_name}_{particles_per_course_bead}_nuc_beads_bonds.txt', 'w') as f:
        for key, value in course_bead_idx_pair_dict.items():
            try:
                f.write(f'({value[0]}, {value[1]}): {nucs[value[0]]} {nucs[value[1]]}\n')
            except:
                f.write(f'({value[0]}): {nucs[value[0]]}\n')
            
    return course_bead_idx_pair_dict, ordered_indexes


def write_sanity_check(coarse_particles_nucleotides, system_name):
    sanity_check = deepcopy(coarse_particles_nucleotides)
    sanity_check = {f'{key}':value for key, value in sanity_check.items()}
    stringify = js.dumps(sanity_check)
    stringify = stringify.split(', "(')
    stringify = ',\n "('.join(stringify)
    with open(f'{system_name}_sanity_check.json', 'w') as f:
        f.write(stringify)


def write_annamo_topology(five_to_three_formatted_list, system_name, particles_per_course_bead):
    x_idx = [i for i,e in enumerate(five_to_three_formatted_list) if e=='x']
    n_beads_in_each_strand = [x_idx[i+1]-x_idx[i]-1 for i in range(len(x_idx)-1)]
    number_of_strands = len(n_beads_in_each_strand)

    with open(f'{system_name}_{particles_per_course_bead}_ANNaMo.top', 'w') as topo_file:
        topo_file.write("{} {}\n".format(sum(n_beads_in_each_strand), number_of_strands))
        idx = 0
        for ctrl,n in enumerate(n_beads_in_each_strand):
            if ctrl == 0:
                typ = 0
            else: typ = 1
            for i in range(n):
                if i == 0 or i == n-1:
                    topo_file.write("{} {} 1\n".format(idx, idx+1+ctrl))
                    if i == 0:
                        topo_file.write("{}\n".format(idx+1))
                    elif i == n-1:
                        topo_file.write("{}\n".format(idx-1))
                else:
                    topo_file.write("{} {} 2\n".format(idx, idx+1+ctrl))
                    topo_file.write("{} {}\n".format(idx-1, idx+1))

                idx += 1

            topo_file.write("\n")


def write_annamo_interaction_matrix(beads_sequences, material):
    beads_sequences = check_material(beads_sequences, material)
    H,S = interaction_matrix(beads_sequences, material, 1)
    x_idx = [i for i,e in enumerate(beads_sequences) if e=='x']

    with open("dHdS_annamo.dat","w") as f_matrix:
        for i in range(1, H.shape[0]-1):
            for j in range(i, H.shape[1]-1):
                if i not in x_idx and j not in x_idx:
                    dH_init = 0
                    dS_init = 0

                    multi_strand_system = len(x_idx) > 2

                    if multi_strand_system:
                        dH_init, dS_init = initiation_contrib(i, j, x_idx)

                    terminal_info = check_terminal(beads_sequences, i, j, x_idx) 
                    if terminal_info["is_terminal"]:
                        dH_terminal, dS_terminal = terminal_penalties(beads_sequences, i, j, terminal_info["condition"], material)
                    else:
                        dH_terminal = 0
                        dS_terminal = 0

                    dH_tot = np.round(H[i][j] + dH_init + dH_terminal, 2)
                    dS_tot = np.round(S[i][j] + dS_init + dS_terminal, 2)

                    f_matrix.write("dH[{}][{}] = {}\n".format(i, j, dH_tot))
                    f_matrix.write("dS[{}][{}] = {}\n".format(i, j, dS_tot))


########################################################
#### Functions to calculate the interaction matrix #####
########################################################

def dH_dS(seq1, seq2):
    dH = 0
    dS = 0
    weight = 0.5

    for k in range(len(seq1)-1):
        if seq1[k:k+2]+seq2[::-1][k:k+2] in dH_stack.keys():
            if k == 0 or k == len(seq1)-2:
                weight = 0.5
                dH += weight * dH_stack[ seq1[k:k+2]+seq2[::-1][k:k+2] ]
                dS += weight * dS_stack[ seq1[k:k+2]+seq2[::-1][k:k+2] ]

            else:
                weight = 1
                dH += dH_stack[ seq1[k:k+2]+seq2[::-1][k:k+2] ]
                dS += dS_stack[ seq1[k:k+2]+seq2[::-1][k:k+2] ]

        else:
            pass

    return dH, dS


def sliding_window(seqs, i, j):
    slide_time = len(seqs[i]) - len(seqs[j])
    HS = []
    seq_j = seqs[j+1][0]+seqs[j][::-1]+seqs[j-1][-1]
    for sl in range(slide_time + 1):
        if sl == 0:
            if sl == slide_time:
                HS.append( dH_dS(seqs[i-1][-1]+seqs[i]+seqs[i+1][0],  seq_j[::-1] ) )
            else:
                HS.append( dH_dS(seqs[i-1][-1]+seqs[i][:len(seqs[j])+1],  seq_j[::-1] ) )

        elif sl == slide_time:
            HS.append( dH_dS(seqs[i][sl-1:sl+len(seqs[j])]+seqs[i+1][0],  seq_j[::-1] ) )

        else:
            HS.append( dH_dS(seqs[i][sl-1:sl+len(seqs[j])+1], seq_j[::-1] ) )

    HS.sort(key=lambda x: x[0])

    return HS[0][0], HS[0][1]


def ordering_seqs(seq1, seq2, idx1, idx2):
    if len(seq1)>=len(seq2):
        return idx1, idx2
    else:
        return idx2, idx1


def interaction_matrix(seqs,material,salt_conc):
    dH = np.zeros((len(seqs),len(seqs)))
    dS = np.zeros((len(seqs),len(seqs)))

    for i in range(1, len(seqs)-1):
        for j in range(i+1, len(seqs)-1):

            longer, shorter = ordering_seqs(seqs[i], seqs[j], i, j)

            H, S = sliding_window(seqs, longer, shorter)
            dH[i][j] = H
            dS[i][j] = S

            #salt_correction = 0.368 * (len(seqs[i]) - 1.0)  * math.log(salt_conc)
            #dS[i][j] += salt_correction

    return dH, dS


def initiation_contrib(i, j, x_idx):
    lower_x = [e for e in x_idx if e<i][-1]     
    upper_x = [e for e in x_idx if e>i][0] 

    ij_not_same_strand = j > upper_x
    if ij_not_same_strand:
        size_strand_i = upper_x - lower_x - 1
        lower_x_j = [e for e in x_idx if e<j][-1]
        upper_x_j = [e for e in x_idx if e>j][0]
        size_strand_j = upper_x_j - lower_x_j - 1

        dH_init = dH_initiation/min(size_strand_i, size_strand_j)
        dS_init = dS_initiation/min(size_strand_i, size_strand_j)

        return dH_init, dS_init
    
    else:
        return 0, 0

def check_terminal(tris, i, j, x_idx):
    i_is_initial = i in [e+1 for e in x_idx]
    j_is_final = j in [e-1 for e in x_idx]
    terminal_1 = tris[i][0]+tris[j][-1]
    condition_1 = i_is_initial and j_is_final and terminal_1 in dH_terminal_penalties.keys()

    i_is_final = i in [e-1 for e in x_idx]
    j_is_initial = j in [e+1 for e in x_idx]
    terminal_2 = tris[j][0]+tris[i][-1]
    condition_2 = i_is_final and j_is_initial and terminal_2 in dH_terminal_penalties.keys()

    cond = "condition_i_init_j_fin" if condition_1 else "condition_i_fin_j_init" if condition_2 else None

    terminal_info = {
        "is_terminal": i_is_final and j_is_initial or i_is_final and j_is_initial,
        "condition": cond
        }
    
    return terminal_info


def terminal_penalties(beads_sequences, i, j, condition, material):
    if condition == "condition_i_init_j_fin":
        terminal = beads_sequences[i][0] + beads_sequences[j][-1]
        if material == "DNA":
            dH_term = dH_terminal_penalties[terminal]
            dS_term = dS_terminal_penalties[terminal]
            return dH_term, dS_term

        else:
            pre_terminal = beads_sequences[1][0] + beads_sequences[j][-2]
            if pre_terminal in dH_terminal_penalties[terminal]:
                dH_term = dH_terminal_penalties[terminal][pre_terminal]
                dS_term = dS_terminal_penalties[terminal][pre_terminal]
                return dH_term, dS_term
            else:
                return 0, 0
            
    elif condition == "condition_i_fin_j_init":
        terminal = beads_sequences[j][0] + beads_sequences[i][-1]
        if material == "DNA":
            dH_term = dH_terminal_penalties[terminal]
            dS_term = dS_terminal_penalties[terminal]
            return dH_term, dS_term

        else:
            pre_terminal = beads_sequences[j][1] + beads_sequences[i][-2]
            if pre_terminal in dH_terminal_penalties[terminal]:
                dH_term = dH_terminal_penalties[terminal][pre_terminal]
                dS_term = dS_terminal_penalties[terminal][pre_terminal]
                return dH_term, dS_term
            else:
                return 0, 0


def check_material(beads_sequences, material):
    material = material.upper()
    if material == 'DNA':
        beads_sequences = [beads_sequences[i].upper().replace('U','T') for i in range(len(beads_sequences))]
        from DNA_SL import dH_stack, dS_stack, dH_initiation, dS_initiation, dH_terminal_penalties, dS_terminal_penalties
    elif material =='RNA':
        beads_sequences = [beads_sequences[i].upper().replace('T','U') for i in range(len(beads_sequences))]
        from RNA_22 import dH_stack, dS_stack, dH_initiation, dS_initiation, dH_terminal_penalties, dS_terminal_penalties
    else:
        print("Error: material option must be specified! Choose between RNA and DNA")
        exit(0)

    global dH_stack
    global dH_initiation
    global dH_terminal_penalties
    global dS_stack
    global dS_initiation
    global dS_terminal_penalties

    return beads_sequences
  
  
########################################################
#### Depreciated functions #############################
########################################################
  
def write_course_particle_files(coarse_particles_nucleotides, coarse_particles_positions, coarse_particle_indexes, system_name):
    sanity_check = deepcopy(coarse_particles_nucleotides)

    indexes_ordered_acending = []
    for values in coarse_particle_indexes.values():
        indexes_ordered_acending.append(values)
    indexes_ordered_acending.sort()
    
    keys_ordered_acending = []
    for lists in indexes_ordered_acending:
        for keys, values in coarse_particle_indexes.items():
            if lists == values:
                keys_ordered_acending.append(keys)
                
    for idx in range(int(len(coarse_particles_nucleotides)/2)):
        for lists in range(len(coarse_particles_nucleotides[(idx+1,1)])):
            coarse_particles_nucleotides[(idx+1,1)][lists] = coarse_particles_nucleotides[(idx+1,1)][lists][::-1]

    for idx in range(int(len(coarse_particles_nucleotides)/2)):
        coarse_particles_nucleotides[(idx+1,1)] = coarse_particles_nucleotides[(idx+1,1)][::-1]

    coarse_particles_nucleotides_ordered = [coarse_particles_nucleotides[order] for order in keys_ordered_acending]

    for idx in range(int(len(coarse_particles_positions)/2)):
        for lists in range(len(coarse_particles_positions[(idx+1,1)])):
            coarse_particles_positions[(idx+1,1)][lists] = coarse_particles_positions[(idx+1,1)][lists][::-1]

    for idx in range(int(len(coarse_particles_positions)/2)):
        coarse_particles_positions[(idx+1,1)] = coarse_particles_positions[(idx+1,1)][::-1]

    coarse_particles_positions_ordered = [coarse_particles_positions[order] for order in keys_ordered_acending]

    nuc_beads_dic = []
    for lists in coarse_particles_nucleotides_ordered:
        for string in lists:
            nuc_beads_dic.append(string)


    nuc_beads_dic = {f'{key}':string for key, string in enumerate(nuc_beads_dic)}

    with open(f'{system_name}_4_nuc_beads_sequence.txt', 'w') as f:
        for key, value in nuc_beads_dic.items():
            f.write(f'{key} {value}\n')

    with open(f'{system_name}_coarse_particles_positions.xyz', 'w') as file:
        for bead_positions in coarse_particles_positions_ordered:
            for position in bead_positions:
                # Write x, y, z positions separated by spaces
                file.write(f"{position[0]} {position[1]} {position[2]}\n")

    sanity_check = {f'{key}':value for key, value in sanity_check.items()}
    stringify = js.dumps(sanity_check)
    stringify = stringify.split(', "(')
    stringify = ',\n "('.join(stringify)
    with open(f'{system_name}_sanity_check.json', 'w') as f:
        f.write(stringify)
        
    for idx in range(int(len(coarse_particle_indexes)/2)):
        for lists in range(len(coarse_particle_indexes[(idx+1,1)])):
            coarse_particle_indexes[(idx+1,1)][lists] = coarse_particle_indexes[(idx+1,1)][lists][::-1]
            
    for idx in range(int(len(coarse_particle_indexes)/2)):
        coarse_particle_indexes[(idx+1,1)] = coarse_particle_indexes[(idx+1,1)][::-1]
    
    pair_dict = {}
    i = 0
    for (s, values) in list(coarse_particle_indexes.items())[::2]:
        for bead in values:
            pair_dict[i] = [bead]
            i += 1
            
    i = 0
    for (s, values) in list(coarse_particle_indexes.items())[1::2]:
        for bead in values:
            pair_dict[i].append(bead)
            i += 1
    
    ordered_indexes = []
    for lists in indexes_ordered_acending:
        for embedded_list in lists:
            ordered_indexes.append(embedded_list)
            
    ordered_indexes = {f'{key}':ind for key,ind in enumerate(ordered_indexes)}
    
    dict_keys = list(ordered_indexes.keys())
    dict_values = list(ordered_indexes.values())
    
    nuc_beads_dic = []
    for lists in coarse_particles_nucleotides_ordered:
        for string in lists:
            nuc_beads_dic.append(string)
            
    bead_pair_dict = {}
    i = 0
    for beads in pair_dict.values():
        bead_pair_dict[i] = (dict_values.index(beads[0]), dict_values.index(beads[1]))
        i +=1
        
    with open(f'{system_name}_4_nuc_beads_bonds.txt', 'w') as f:
        for key, value in bead_pair_dict.items():
            f.write(f'{value[0]} {value[1]}\n')

    return None


@contextmanager
def suppress_output():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        # Backup the original stdout and stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            # Redirect both stdout and stderr to devnull
            sys.stdout, sys.stderr = fnull, fnull
            yield
        finally:
            # Restore stdout and stderr to their original values
            sys.stdout, sys.stderr = old_stdout, old_stderr

        
        
        
def fully_complementary_sequential_fix(nucleotides_in_duplex, positions, all_edge_cases, duplex_to_particle):
    pair_set, indexes, nearest_edge_case_min = full_sequential_group_distance_check(positions, all_edge_cases)

    easy_fix = []
    monomer_info = np.array(nucleotides_in_duplex)
    for idxes in indexes:
        nuc_dict = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
        idx_1 = idxes[0]
        idx_2 = idxes[1]

        monomers_1 = monomer_info[idx_1]
        monomers_1 = [mono.btype for mono in monomers_1]

        monomers_2 = monomer_info[idx_2]
        monomers_2 = [nuc_dict[mono.btype] for mono in monomers_2][::-1]

        if monomers_1 == monomers_2:
            new_key = len(duplex_to_particle) + 1
            duplex_to_particle[int(new_key)] = [idxes[0], idxes[1][::-1]]
            easy_fix.append([idxes[0], idxes[1][::-1]])
            

    # for key, value in duplex_to_particle.items():
    #     print(value)
    #     ends = value[0][-1]
    #     starts = value[0][0]
    #     for pairs in  easy_fix:
    #         look_0 = pairs[0][0] - 1
    #         look_1 = pairs[0][0] + 1
    #         if ends == look_0:
    #             key_int = int(key) +1
    #             key_mapping_base = {str(key):(key) for key in range(key_int +1)}
    #             key_mapping_update = {str(key):str(key+1) for key in range(key_int, len(duplex_to_particle) +1)}
    #             key_mapping_base.update(key_mapping_update) 

    #             old_duplex = {key_mapping_base.get(str(key), str(key)): value for key, value in duplex_to_particle.items() if int(key) >= key_int}

    #             duplex_to_particle = {key_mapping_base.get(str(key), str(key)): value for key, value in duplex_to_particle.items() if int(key) < key_int}
    #             duplex_to_particle[str(key_int)] = pairs
    #             duplex_to_particle.update(old_duplex)
    #             print(pairs)

        
    # easy_fix = np.concatenate(np.concatenate(np.array(easy_fix, dtype="object")))
    
    return duplex_to_particle, easy_fix