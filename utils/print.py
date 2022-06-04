


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\x1b[94m'
    OKCYAN = '\033[36m'
    OKPURPLE = '\x1b[95m'
    OKGREEN = '\x1b[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_info(*args, end="\n"):
    print(bcolors.OKGREEN + " ".join(map(str, args)) + bcolors.ENDC, end=end)

def print_error(*args, end="\n"):
    print(bcolors.FAIL + " ".join(map(str, args)) + bcolors.ENDC)

def print_blue(*args, end="\n"):
    print(bcolors.OKBLUE + " ".join(map(str, args)) + bcolors.ENDC, end=end)