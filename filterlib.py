import cv2

#=============================================================================================
# Call this function if selected filterName is not defined
#=============================================================================================
def filterNotDefined(inputImage,args=None):
    print('Filter name not defined in filterlib.py')
    return inputImage


def grey(inputImage,args=None):
    """Convert a colour image to greyscale.

    Usage: grey()
    """
    if len(inputImage.shape) == 3:
        return cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
    else:
        return inputImage


def color(inputImage,args=None):
    """Convert a greyscale image to colour.

    Usage: color()
    """
    if len(inputImage.shape) == 2:
        return cv2.cvtColor(inputImage, cv2.COLOR_GRAY2BGR)
    else:
        return inputImage


def blur(inputImage,args):
    """Perform a Gaussian blur on an image.

    Usage: blur(n) where n in an integer
    Since only odd numbers are allowed in Gaussian blur, we use 2*n+1
    as the radius.
    """
    arg = args.split(',')
    return cv2.GaussianBlur(inputImage,(int(arg[0])*2+1,int(arg[0])*2+1),0)


def threshold(inputImage,args):
    """Perform a binary threshold on a greyscale image.

    Usage: threshold(lowerBound, higherBound)
    """
    arg = args.split(',')
    _, ret = cv2.threshold(
        inputImage, int(arg[0]), int(arg[1]), cv2.THRESH_BINARY
        )
    return ret


def canny(inputImage,args):
    """Extract canny edges from a greyscale image.

    Usage: canny(minVal, maxVal)
    """
    arg = args.split(',')
    return cv2.Canny(inputImage,int(arg[0]),int(arg[1]))


def erode(inputImage,args):
    """Perform erosion on a binary image.

    Usage: erode(kernelSize)
    """
    arg = args.split(',')
    kernel = np.ones((int(arg[0]),int(arg[0])), np.uint8)
    return cv2.erode(inputImage, kernel, iterations=1)


def dilate(inputImage,args):
    """Perform dilation on a binary image.

    Usage: dilate(kernelSize)
    """
    arg = args.split(',')
    kernel = np.ones((int(arg[0]),int(arg[0])), np.uint8)
    return cv2.dilate(inputImage, kernel, iterations=1)
