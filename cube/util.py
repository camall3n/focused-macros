class ansi: pass
ansi.red = '\x1b[41m'
ansi.orange = '\x1b[48;5;130m'
ansi.green = '\x1b[42m'
ansi.yellow = '\x1b[43m'
ansi.blue = '\x1b[44m'
ansi.magenta = '\x1b[45m'
ansi.cyan = '\x1b[46m'
ansi.white = '\x1b[47m'
ansi.reset = '\x1b[49m'

fill = {
    'O': ansi.orange+' '+ansi.reset,
    'G': ansi.green+' '+ansi.reset,
    'Y': ansi.yellow+' '+ansi.reset,
    'R': ansi.red+' '+ansi.reset,
    'B': ansi.cyan+' '+ansi.reset,
    'W': ansi.white+' '+ansi.reset,
}
