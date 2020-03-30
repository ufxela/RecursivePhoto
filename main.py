from recursivePhoto.recursivePhotoV1 import recursivePhotoV1
from recursivePhoto.recursivePhotoV2 import recursivePhotoV2
from recursivePhoto.recursivePhotoV3 import recursivePhotoV3
from recursivePhoto.recursivePhotoV4 import recursivePhotoV4
from recursivePhoto.recursivePhotoV5 import recursivePhotoV5
from recursivePhoto.recursiveVideo import recursiveVideo

options = [
    (recursivePhotoV1, "recursivePhotoV1 (photo made from itself, b/w, no square)"),
    (recursivePhotoV2, "recursivePhotoV2 (photo made from different photos, b/w, square)"),
    (recursivePhotoV3, "recursivePhotoV3 (photo made from different photos, color, square)"),
    (recursivePhotoV4, "recursivePhotoV4 (photo made from different photos, b/w, square, uses pattern matching)"),
    (recursivePhotoV5, "recursivePhotoV5 (photo made from different photos, color, square, uses pattern matching)"),
    (recursiveVideo, "recursiveVideo (video made from photos made from themselves, b/w, no square)"),
    (quit, "quit")
]

def printopen():
    print("--------------------------")
    print("Welcome to Recursive Photo")
    print("--------------------------\n\n")
    print("Options:")
    for i in range(len(options)):
        print(str(i) + ": " + options[i][1])
    print("\nInput the number (e.g. 0) of which program you want to run:\n")

def quit():
    print("Quitting. Goodbye.")

if __name__ == '__main__':
    program_index = 0
    printopen()
    while 1:
        try:
            program_index = int(input())
            if(program_index < 0 or program_index >= len(options)):
                print("You did not make a valid selection. Try again.")
                continue
            break
        except:
            print("You did not type in an integer. Try again.")

    print("Executing option " + str(program_index))
    options[program_index][0]()
    print("All done! Program Exiting.")