import subprocess

def rsync(source, dest):
    args = ["rsync", "-zurP", source, dest]
    print(' '.join(args))
    subprocess.run(args, env={'LANG': 'en_US.UTF-8'})

class ansi_colors: pass
ansi_colors.red = '\x1b[41m'
ansi_colors.orange = '\x1b[48;5;130m'
ansi_colors.green = '\x1b[42m'
ansi_colors.yellow = '\x1b[43m'
ansi_colors.blue = '\x1b[44m'
ansi_colors.magenta = '\x1b[45m'
ansi_colors.cyan = '\x1b[46m'
ansi_colors.white = '\x1b[47m'
ansi_colors.reset = '\x1b[49m'
