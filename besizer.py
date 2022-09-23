
import cv2
import numpy as np
from scipy.special import comb

def extract_contours(image):
    # rgb->gray
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gaussian filter
    gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    # binary exp-threshold=0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # find contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a matrix of numpy
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]-
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints, yPoints = points[:,0], points[:,1]
    bezier_out = []
    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
 
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    bezier_out = [[[xvals[i], yvals[i]]] for i in range(len(xvals))]
    bezier_out = np.array(bezier_out, dtype=np.int32)

    return bezier_out


if __name__ == '__main__':

    import copy
    img = cv2.imread("results/B端24/test_unknown_style_latest/images/AaJueXingHei60J/0.png")
    img1 = copy.deepcopy(img)
    img2 = copy.deepcopy(img)
    contours = extract_contours(img)
    color_map = [(0,0,255), (0,255,0),(255,0,0),(255,255,0)]

    # 原始轮廓
    for i in range(len(contours)):
      cv2.drawContours(img, contours, i, color_map[i%len(color_map)], 5)
    cv2.imwrite('draw.jpg', img)

    ## cv2.approxPolyDP 拟合
    for i, cont in enumerate(contours):
      approx = cv2.approxPolyDP(cont, 5, True)
      cv2.drawContours(img1, approx, -1, color_map[i%len(color_map)], 5)
    cv2.imwrite('draw1.jpg', img1)

     ## 贝塞尔曲线拟合
    for i, cont in enumerate(contours):
        cont0 = cont.squeeze()
        approx = bezier_curve(cont0, nTimes=20)
        cv2.drawContours(img2, approx, -1, color_map[i%len(color_map)], 5)
    cv2.imwrite('draw2.jpg', img2)
