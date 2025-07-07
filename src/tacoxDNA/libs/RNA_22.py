# written in SantaLucia notation: ATTA=AT/TA ,
# 5'-AT-3' / 3'-TA-5'
dH_stack = {
        'GCCG': -16.52,
        'CCGG': -13.94,
        'GACU': -13.75,
        'CGGC': -9.61,
        'ACUG': -11.98,
        'CAGU': -10.47,
        'AGUC': -9.34, 
        'UAAU': -9.16,
        'AUUA': -8.91,
        'AAUU': -7.44,
        'GCUG': -14.73,
        'CUGG': -9.26,
        'GGCU': -12.41,
        'CGGU': -5.64,
        'AUUG': -9.23,
        'GAUU': -10.58,
        'UGGU': -8.76,
        'UAGU': -2.72,
        'GGUU': -9.06,
        'GUUG': -7.66,
        'AGUU': -5.1,
        'GGCC': -13.94,
        'UCAG': -13.75,
        'CUGA': -9.34,
        'GUCA': -11.98,
        'UGAC': -10.47,
        'UUAA': -7.44,
        'GUCG': -14.73,
        'UGGC': -5.64,
        'GGUC': -9.26,
        'UCGG': -12.41,
        'GUUA': -9.23,
        'UGAU': -2.72,
        'UUAG': -10.58,
        'UUGA': -5.1,
        'UUGG': -9.06
}
dS_stack = {
        'GCCG': -42.13,
        'CCGG': -34.41,
        'GACU': -36.53,
        'CGGC': -23.46,
        'ACUG': -31.37,
        'CAGU': -27.08,
        'AGUC': -23.66,
        'UAAU': -25.4,
        'AUUA': -25.22,
        'AAUU': -20.98,
        'GCUG': -40.32,
        'CUGG': -23.64,
        'GGCU': -34.23,
        'CGGU': -14.83,
        'AUUG': -27.32,
        'GAUU': -32.19,
        'UGGU': -27.04,
        'UAGU': -8.08,
        'GGUU': -28.57,
        'GUUG': -24.11,
        'AGUU': -16.53,
        'GGCC': -34.41,
        'UCAG': -36.53,
        'CUGA': -23.66,
        'GUCA': -31.37,
        'UGAC': -27.08,
        'UUAA': -20.98,
        'GUCG': -40.32,
        'UGGC': -14.83,
        'GGUC': -23.64,
        'UCGG': -34.23,
        'GUUA': -27.32,
        'UGAU': -8.08,
        'UUAG': -32.19,
        'UUGA': -16.53,
        'UUGG': -28.57
        }
dH_initiation = 4.66
dS_initiation = 1.78
dH_terminal_penalties = {
    'AU': {
        'UA': 4.36,
        'GC': 3.17,
        'UG': 5.16
    },
    'UA': {
        'AU': 4.36,
        'CG': 3.17,
        'GU': 5.16
    },
    'GU': {
        'UA': 3.65,
        'GC': 3.91,
        'UG': 6.23
    },
    'UG': {
        'AU': 3.65,
        'CG': 3.91,
        'GU': 6.23
    }
}
dS_terminal_penalties = {
    'AU': {
        'UA': 13.35,
        'GC': 8.79,
        'UG': 18.96
    },
    'UA': {
        'AU': 13.35,
        'CG': 8.79,
        'GU': 18.96
    },
    'GU': {
        'UA': 12.78,
        'GC': 12.17,
        'UG': 22.47
    },
    'UG': {
        'AU': 12.78,
        'CG': 12.17,
        'GU': 22.47
    }
}
