import torch

class bcolors:
    """A class with some basic colors."""

    #####################################

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def welcome(ARGS):
    """A welcom method that checks wether cuda is available and prints out the given Arguments

    Parameters:
    -----------
    ARGS: dict of Arguments
        arguments that should be printet

    """

    if torch.cuda.is_available():
        print("Your GPU can be used by Torch")
    print('\n' +  f"{bcolors.WARNING}Torch does not use your GPU. Check if CUDA supports your GPU or update the GPU driver. \nCUDA is not essentially needed but training your models will take more time!{bcolors.ENDC}")
    print('The used parameters:')
    for i in ARGS.__dict__:
        print(str(i) + ": " + str(ARGS.__dict__[i]))

def hitground(ARGS, pos):
    """Checks wether the ground was hit and prints out the position.

    Parameters:
    -----------
    ARGS: dict of Arguments
        arguments that should be printet
    pos: List[float]
        the current position
    """
    if (not(pos[2] >= 0.075 and ARGS.target_z > 0.1)):
        print('\n' +  f"{bcolors.WARNING}You hit the ground. \nAny Problems with flying Ducks?{bcolors.ENDC}")
    else:
        print("Done! Endposition:")
    print('Pos: ' + str(pos))

def debug(color: str, message: str):
    """Prints out the message in a color.

    Parameters:
    -----------
    color: str
        the color of the print
    message: str
        the message of the print
    """
    string = f"{color}" + message
    print('\n' + string)
    print(f"{bcolors.WARNING}{bcolors.ENDC}")



