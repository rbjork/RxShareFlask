from __future__ import division
from sympy.interactive import printing
from sympy import *
from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage
import cv2
import math
import sympy as sym
from sympy.utilities.lambdify import lambdify
from alignAndStitchImagesRansac import AlignAndStitchImagesRansac


def loadimages():
    pass



def edges():
    pass


## CELL
def getMidline(params,bottle):  # Returns slope intercept coefficients of bounding lines and midline
    height = bottle.shape[0]
    width = bottle.shape[1]
    rho1 = params[0]
    theta1 = params[1]
    rho2 = params[2]
    theta2 = params[3]
    
    a = np.cos(theta1)
    b = np.sin(theta1)
    x0 = a*rho1
    y0 = b*rho1
    x1 = int(x0 + height*(-b))
    y1 = int(y0 + height*(a))
    x2 = int(x0 - height*(-b))
    y2 = int(y0 - height*(a))
    
    cv2.line(bottle,(x1,y1),(x2,y2),(0,0,255),2)
    
    mx1 = (x2 - x1)/float(y2 - y1)
    bx1 = x2 - mx1*y2
 #   print "mx1 calc",x1,x2,y1,y2,mx1
    
    a = np.cos(theta2)
    b = np.sin(theta2)
    x0 = a*rho2
    y0 = b*rho2
    x1 = int(x0 + height*(-b))
    y1 = int(y0 + height*(a))
    x2 = int(x0 - height*(-b))
    y2 = int(y0 - height*(a))
    
    cv2.line(bottle,(x1,y1),(x2,y2),(0,0,255),2)
    
    mx2 = (x2 - x1)/float(y2 - y1)
    bx2 = x2 - mx2*y2
    
    mx = (mx1 + mx2)/2  # slope between
    bx = (bx1 + bx2)/2  # intercept with x axis
    xm = mx*height + bx
    
#    print "point slope ",bx1,mx1,bx2,mx2
    cv2.line(bottle,(int(bx),0),(int(xm),height),(255,0,255),2) 
    
    return (mx1,bx1, mx2, bx2, mx, bx)

def houghSides(bottle,edges, threshold, left):
    lines = cv2.HoughLines(edges,1,np.pi/180,threshold)
    if lines == None:
        return (0,0)
    cnt = 0
    rholast = 0
    thetalast = 0
    if left:
        for rho,theta in lines[0]:
            if np.abs(theta - 3.14) < 0.1: # It could be that it'll always be the first if they're sorted by votes
                cnt = cnt + 1
                rholast = rho
                thetalast = theta
                break
    else:
        for rho,theta in lines[0]:
            if np.abs(theta) < 0.1:  # It could be that it'll always be the first if they're sorted by votes
                cnt = cnt + 1
                rholast = rho
                thetalast = theta
                break
    return (cnt,rholast,thetalast)
                       
def houghManage(bottle,edges, thresholdinit):
    threshold = thresholdinit
    data = houghSides(bottle,edges, threshold, True)
    cnt = data[0]
    trys = 0
    while cnt == 0 and trys < 10  :  # may be able to use one pass if thershold low and they're sorted
        if cnt > 1:
            threshold = threshold + 5
            data = houghSides(bottle,edges, threshold, True)
        else:
            threshold = threshold - 5
            data = houghSides(bottle,edges, threshold, True)
        cnt = data[0]
       # print cnt
        trys = trys + 1
     #   print totalcnt, threshold
        
    rhoL = data[1]
    thetaL = data[2]
 #   print trys
    #drawHoughLines(bottle, rhoL, thetaL)
   
    threshold = thresholdinit
    data = houghSides(bottle,edges, threshold, False)
    cnt = data[0]
    trys = 0
    
    while cnt ==0 and trys < 10  :
        if cnt > 1:
            threshold = threshold + 5
            data = houghSides(bottle,edges, threshold, False)
        else:
            threshold = threshold - 5
            data = houghSides(bottle,edges, threshold, False)
        cnt = data[0]
        trys = trys + 1
       # print totalcnt, threshold
        
    rhoR = data[1]
    thetaR = data[2]

    return (rhoL,thetaL,rhoR,thetaR)
    


## CELL
def findMidlineEllipseIntercept(edges,boundlines,bottle):
    mx = boundlines[4]
    bx = boundlines[5]
    height = edges.shape[0] # row count or height
    print "height",height
    yrange = np.asarray(range(0,height))
    xfloat = mx*yrange + bx
    xint = xfloat.astype(int)
    pts = edges[yrange,xint]
    ellipseIntercepts = np.where(pts != 0)[0]
    index = ellipseIntercepts[-1:][0]
    x0 = xint[index]; y0 = yrange[index]
    print "findMidlineEllipseIntercept",x0,y0
    cv2.circle(bottle, (x0,y0), 10, (0,127,0),8)
    return (x0,y0)
    


def cot(angle):
    return 1/math.tan(angle)

def acot(x):
    return (math.pi/2 - math.atan(x))

# Either sensorSize and focalLength are in units of pixels or they are in units of length millimeters (mm).
def getFOV(sensorSize,focalLength):
    fovW = 2*math.atan(0.5*sensorSize[0]/focalLength)*(180/math.pi)
    fovH = 2*math.atan(0.5*sensorSize[1]/focalLength)*(180/math.pi)
    return (fovW,fovH)

# pixelToAngle returns tuple of angles in units of degrees
def pixelToAngle(pix,focalLength,sensorRes,density,UnitsPix):  
    pixX = pix[0] - sensorRes[0]/2
    pixY = pix[1] - sensorRes[1]/2
    if UnitsPix == False:
        focalLength = focalLength*density
    angleX = math.atan2(pixX,focalLength)
    angleY = math.atan2(pixY,focalLength)
    return (angleX,angleY)

def angleToPixel(anglePitch,angleYaw,focalLength,sensorRes,density,UnitsPix):
    if UnitsPix == False:
        focalLength = focalLength*density
    print "in angleToPixel sensorRes = ",sensorRes
    #pix_x = focalLength*math.tan(angleYaw) + sensorRes[0]/2
    #pix_y = focalLength*math.tan(anglePitch) + sensorRes[1]/2
    pix_x = focalLength*math.tan(angleYaw) + sensorRes[1]/2
    pix_y = focalLength*math.tan(anglePitch) + sensorRes[0]/2
    return (pix_x,pix_y)




def computeTheta(boundlines,focalLengthPixels,sensorRes):
    mx1 = boundlines[0]
    bx1 = boundlines[1]
    mx2 = boundlines[2]
    bx2 = boundlines[3]
    
    yt = 200
    yb = 500
    
    b, notUsed = pixelToAngle((yt,480),focalLengthPixels,sensorRes,None,True) # chard b
    a, notUsed = pixelToAngle((yb,480),focalLengthPixels,sensorRes,None,True) # chord a
    b = math.pi/2 - b
    a = math.pi/2 - a
    
    print "alphas",b*(180/math.pi),a*(180/math.pi)
    xct = mx1*yt + bx1
    xdt = mx2*yt + bx2
    xcb = mx1*yb + bx1
    xdb = mx2*yb + bx2
    
    G = abs((xdb-xcb)/(xdt-xct))  # chord B / chord A
    print "G",G
    m = math.sin(b)/math.sin(a)
    factor = (G/m)
    print "factor",factor
    theta = math.atan2((math.sin(a) - factor*math.sin(b)),(math.cos(a) - factor*math.cos(b)))
    return theta

## REVISED - check that value is around 3 to 4. Older version was giving 1.3 - not likely correct
def computeHlensRadRatio(beta,phi0,thetaX):
    r = 1
    d = r/math.sin(phi0) - r
    hlen = d/math.tan(beta - thetaX)
    return hlen

## REVISED
def computeZeta(beta,phi0,thetaX):
    r = 1
    d = r/math.sin(phi0) - r
    print "d",d
    hlen = d/math.tan(beta - thetaX)
    zeta = math.atan(hlen/(r+d))
    return zeta

### REVISED
def computePhi0(boundlines,sensorRes,thetaX,focalLength):
    pix_x, pix_y = angleToPixel(thetaX,0,focalLength,sensorRes,None,True)
    mx1 = boundlines[0]
    bx1 = boundlines[1]
    mx2 = boundlines[2]
    bx2 = boundlines[3]
    xct = mx1*pix_y + bx1
    xdt = mx2*pix_y + bx2
    pxchord = abs(xct-xdt)/2 + sensorRes[1]/2
    notUsed, phi0  = pixelToAngle((1,pxchord),focalLengthPixels,sensorRes,None,True)
    return phi0

def computeThetaZ(boundlines):
    mx = boundlines[4]
    return math.atan(mx)
    
#def computeAlpha0(ellipseParams,sensorRes,focalLengthPixels):
#    xctr, yctr, major, minor = ellipseParams
#    notUsed, atemp = pixelToAngle((xctr,yctr), focalLengthPixels,sensorRes, None, True)
#    alpha0 = math.pi/2 - atemp
#    return alpha0

def computeBeta(x0,y0,sensorRes,focalLengthPixels):
    print "sensorRes",sensorRes
    print "x0,y0",x0,y0
    btemp, notUsed = pixelToAngle((y0,x0), focalLengthPixels,sensorRes, None, True)
    beta = math.pi/2 - btemp
    return beta
    



def computeAffineMapping(boundlines,phi0,x0,y0,thetaX,thetaZ,hlen,zeta,focalLength):
    
    ######  NOTE THE FUDGE FACTOR on xy_multiplier ADDED !!! #########
    
    Z = thetaX + math.pi/2 - zeta 
    print "thetaX,zeta,Z",thetaX,zeta,Z
    Zcam = math.pi/2 - Z
    pix_x, pix_y = angleToPixel(Zcam,0,focalLength,sensorRes,None,True) # this is chord at Z
    print "boundlines",boundlines
    mx1 = boundlines[0]
    bx1 = boundlines[1]
    mx2 = boundlines[2]
    bx2 = boundlines[3]
    print "pix_x, pix_y",pix_x, pix_y
    xct = mx1*pix_y + bx1
    xdt = mx2*pix_y + bx2
    print "xct xdt",xct,xdt
    pxchord = abs(xct-xdt)
    
    ww = bxfunc(0,phi0,thetaX,thetaZ,hlen,zeta)  ## width at h=0(base of bottle)
    print "pxchord ww",pxchord,ww
    xy_multiplier = (pxchord/2)/ww    # This is chord at alpha0  (Not quite chord Z but hopefully close enough)
    xzero = -xy_multiplier*bxfunc(0,math.pi/2,thetaX,thetaZ,hlen,zeta)
    xoffset = x0 - xzero - 2
    
    #yzero = -xy_multiplier*byfunc(0,math.pi/2,thetaX,hlen,zeta)
    yzero = -.95*xy_multiplier*byfunc(0,math.pi/2,thetaX,thetaZ,hlen,zeta)
    
    yoffset = y0 - yzero
    print yzero
    return (xy_multiplier,xoffset,yoffset)
    



def getcolor(ipos,h,bottlerotnd,metric):
    if ipos.is_integer():
        iH = int(ipos + 1);
    else:
        iH = int(math.ceil(ipos))
        
    iL = int(math.floor(ipos))
   
    if h.is_integer():
        hH = int(h + 1)
    else:
        hH = int(math.ceil(h))
        
    hL = int(math.floor(h))

    try:
        px1 = bottlerotnd[iL,hL]
        px2 = bottlerotnd[iH,hL]
        px3 = bottlerotnd[iL,hH]
        px4 = bottlerotnd[iH,hH]

       # px1 = photopixels[iL][hL] # for testing with photopixels = [[(10,20,30),(40,50,60)],[(70,80,90),(100,110,120)]]
       # px2 = photopixels[iH][hL]
       # px3 = photopixels[iL][hH]
       # px4 = photopixels[iH][hH]

        dxH = iH - ipos
        dxL = ipos - iL
        dyH = hH - h
        dyL = h - hL

        if metric == 1:  # L1 norm or Manhattan distance
            d1 = abs(dxL) + abs(dyL)
            d2 = abs(dxH) + abs(dyL)
            d3 = abs(dxL) + abs(dyH)
            d4 = abs(dxH) + abs(dyH)
        else:           # L2 norm or pythagrean distance
            d1 = math.sqrt(dxL**2 + dyL**2)
            d2 = math.sqrt(dxH**2 + dyL**2)
            d3 = math.sqrt(dxL**2 + dyH**2)
            d4 = math.sqrt(dxH**2 + dyH**2)

        if d1 < 0.01:  # so close to pixel that interpolation with other pixel values isn't needed
            return px1
        elif d2 < 0.01: # same thing ...
            return px2
        elif d3 < 0.01:
            return px3
        elif d4 < 0.01:
            return px4

        D = 1/d1 + 1/d2 + 1/d3 + 1/d4  # nomalization factor of wts computation

        wt1 = (1/d1)/D
        wt2 = (1/d2)/D
        wt3 = (1/d3)/D
        wt4 = (1/d4)/D

        red =   wt1*px1[0] + wt2*px2[0] + wt3*px3[0] + wt4*px4[0]
        green = wt1*px1[1] + wt2*px2[1] + wt3*px3[1] + wt4*px4[1]
        blue =  wt1*px1[2] + wt2*px2[2] + wt3*px3[2] + wt4*px4[2]



        return (int(red), int(green), int(blue))

    except IndexError:
        return None



def changeshapeInverseOptics(imgnew, bottlerotnd, phi0, thetaX,thetaZ, mapping, hLenVal, zetaval, aspectRatio):
    xy_multiplier = mapping[0]
    xoffset = mapping[1]
    yoffset = mapping[2]
    nonecount = 0
    testcnt = 0
    print "img size",imgnew.size
    imax = imgnew.size[0]
    jmax = imgnew.size[1]
    pixelsnew = imgnew.load()
    angles = np.linspace(phi0, math.pi-phi0, jmax)
    
    ## NEW
        #dphi = (math.pi - 2*phi0)/jmax
        #dr = rval*dphi  # each pix seperated by dr horizontally
        #delH = dr # So each pix is seperated this same amount vertically - this preserves character aspect ratio
        #height = aspectRatio*math.pi  # this is height as a multiple of r
        #imax = int(height/delH) # this is where we will crop at the top which should be top of bottle
    ## end New
    
    for i in range(imax-1):    # for every pixel. Note that this is slow - using nested loops
       # hval = i*delH
        hval = 2*aspectRatio*rval*(i/imax)
        jphoto = -(xy_multiplier+10*hval)*bxfunc(hval,angles,thetaX,thetaZ,hLenVal,zetaval) + xoffset
        iphoto = -(.95*xy_multiplier-7*hval)*byfunc(hval,angles,thetaX,thetaZ,hLenVal,zetaval) + yoffset # bilinear interpolation is likely better than this
        for j in range(jmax-1):   # try to eliminate this loop by passing arrays to getcolor
            px = getcolor(iphoto[j],jphoto[j], bottlerotnd, 1) #getColorNoInterpolation(ipos,h) #getcolorOld(iphoto[i],jphoto[i], testcnt)
            testcnt = testcnt + 1
            if(px is None ):
                pixelsnew[imax-i-1,j] = (0, 0, 0)
                nonecount = nonecount + 1
            else:
                pixelsnew[i,j] = (px[0], px[1], px[2])
                
    print(nonecount)



# LOAD IMAGES
bottleImage1 = Image.open('./images/bot1striped.jpg','r')
bottle1rot = bottleImage1 
bottleImage2 = Image.open('./images/bot2striped.jpg','r')
bottle2rot = bottleImage2
bottleImage3 = Image.open('./images/bot3striped.jpg','r')
bottle3rot = bottleImage3


# CONVERT TO NDARRAY AND PERFORM EDGE DETECTIION
bottle1rotnd = np.asarray(bottle1rot)
img1 = ndimage.gaussian_filter(bottle1rotnd, 1)
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
edges1 = cv2.Canny(gray1,20,150, apertureSize = 3, L2gradient = True)

bottle2rotnd = np.asarray(bottle2rot)
img2 = ndimage.gaussian_filter(bottle2rotnd, 1)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
edges2 = cv2.Canny(gray2,20,150, apertureSize = 3, L2gradient = True)

bottle3rotnd = np.asarray(bottle3rot)
img3 = ndimage.gaussian_filter(bottle3rotnd, 1)
gray3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
edges3 = cv2.Canny(gray3,20,150, apertureSize = 3, L2gradient = True)

# Redundant
#bottle1rotnd = np.asarray(bottle1rot)    
#bottle2rotnd = np.asarray(bottle2rot)   
#bottle3rotnd = np.asarray(bottle3rot)

# GET HOUGH PARAMS OF LINES
params1 = houghManage(bottle1rotnd,edges1,100)
params2 = houghManage(bottle2rotnd,edges2,100)
params3 = houghManage(bottle3rotnd,edges3,100)

print params1, params2, params3

# GENERATED Affine equations describing the left and right edges and the center
boundlines1 = getMidline(params1,bottle1rotnd) # gets coefficients
boundlines2 = getMidline(params2,bottle2rotnd)
boundlines3 = getMidline(params3,bottle3rotnd)

# Locates the front edge of the base of the bottle
x01, y01 = findMidlineEllipseIntercept(edges1,boundlines1,bottle1rotnd)
x02, y02 = findMidlineEllipseIntercept(edges2,boundlines2,bottle2rotnd)
x03, y03 = findMidlineEllipseIntercept(edges3,boundlines3,bottle3rotnd)





# SYMBOLIC Transform - should replace with optimized cython
x, y, z, h = symbols('x y z h')
d, r, phi, zeta, h_l, beta_s = symbols('d r phi zeta h_l beta')
theta_x, theta_y, theta_z = symbols('theta_x theta_y theta_z')
bx,by = symbols('bx, by')
c_x, c_y, c_z = symbols('c_x c_y c_z')
a_x, a_y, a_z = symbols('a_x a_y a_z')
b_x, b_y = symbols('b_x b_y')

bx = (r*cos(phi)*cos(theta_z) + (h-h_l)*sin(theta_z))/(r*sin(theta_x)*sin(theta_z)*cos(phi)-(h - h_l)*sin(theta_x)*cos(theta_z) + (h_l*cos(zeta)/sin(zeta) + r*(-sin(phi) + 1) - r)*cos(theta_x))
by = (-r*sin(theta_z)*cos(phi)*cos(theta_x) + (h - h_l)*cos(theta_x)*cos(theta_z) + (h_l*cos(zeta)/sin(zeta) + r*(-sin(phi) + 1) - r)*sin(theta_x))/(r*sin(theta_x)*sin(theta_z)*cos(phi)-(h - h_l)*sin(theta_x)*cos(theta_z) + (h_l*cos(zeta)/sin(zeta) + r*(-sin(phi) + 1) - r)*cos(theta_x))

alpha_0, phi_0 = symbols('alpha_0 phi_0')
hl = r*(1-sin(phi_0))/(tan(pi-(theta_x - alpha_0)) - tan(pi - (theta_x - beta_s)))


# FOCAL LENGTH COMPUTATION - Shoud used camera params - not fixed numbers below
# For Android APIs up through 20, use getHorizontalAngleOfView and getVerticalAngleOfView
# Otherwise it may require a lookup table
#sensorRes = (3264,2448)
#sensorRes = (1920,1080)
sensorRes = bottle1rotnd.shape  # if reduced

sensorSize = (4.6, 3.5) # mm
FOVw, FOVh = (60,48) # units of degrees

focalLengthPixels = sensorRes[0]/2*cot(FOVw/2*math.pi/180)
fov = getFOV(sensorRes,focalLengthPixels)


    
# ASSUMPTION. That its measured
thetaX1 = computeTheta(boundlines1,focalLengthPixels,sensorRes)
thetaX2 = computeTheta(boundlines2,focalLengthPixels,sensorRes)
thetaX3 = computeTheta(boundlines3,focalLengthPixels,sensorRes)
'''
i = 0
for boundline in boundlines:
    thetaX[i] = computeTheta(boundline,focalLengthPixels,sensorRes)
    i = i + 1
'''

#alpha01 = computeAlpha0(ellipseParams1,sensorRes,focalLengthPixels)
beta1 = computeBeta(x01,y01,sensorRes,focalLengthPixels)
phi01 = computePhi0(boundlines1,sensorRes,thetaX1,focalLengthPixels)
phi01 = abs(phi01)

#alpha02 = computeAlpha0(ellipseParams2,sensorRes,focalLengthPixels)
beta2 = computeBeta(x02,y02,sensorRes,focalLengthPixels)
phi02 = computePhi0(boundlines2,sensorRes,thetaX2,focalLengthPixels)
phi02 = abs(phi02)

#alpha03 = computeAlpha0(ellipseParams3,sensorRes,focalLengthPixels)
beta3 = computeBeta(x03,y03,sensorRes,focalLengthPixels)
phi03 = computePhi0(boundlines3,sensorRes,thetaX3,focalLengthPixels)
phi03 = abs(phi03)

hval1 = computeHlensRadRatio(beta1,phi01,thetaX1)
zetaval1 = computeZeta(beta1, phi01, thetaX1)
Z1 = (thetaX1 + math.pi/2) - zetaval1

hval2 = computeHlensRadRatio(beta2,phi02,thetaX2)
zetaval2 = computeZeta(beta2, phi02, thetaX2)
Z2 = (thetaX2 + math.pi/2) - zetaval2

hval3 = computeHlensRadRatio(beta3,phi03,thetaX3)
zetaval3 = computeZeta(beta3, phi03, thetaX3)
Z3 = (thetaX3 + math.pi/2) - zetaval3

'''
for i in range(numphotos):
   beta[i] = computeBeta(x0[i],y0[i],sensorRes,focalLengthPixels)
   phi0[i] = computePhi0(boundlines[i],sensorRes,thetaX[i],focalLengthPixels)
   phi0[i] = abs(phi0[i])
   hval[i] = computeHlensRadRatio(beta[i],phi0[i],thetaX[i])
   zetaval[i] = computeZeta(beta[i], phi0[i], thetaX[i])
   Z[i] = (thetaX[i] + math.pi/2) - zetaval[i]
'''


thetaZ1 = computeThetaZ(boundlines1)
thetaZ2 = computeThetaZ(boundlines2)
thetaZ3 = computeThetaZ(boundlines3)

rval = 1

bxinstance = bx.subs([(r,rval)])
byinstance = by.subs([(r,rval)])

bxfunc = lambdify((h, phi, theta_x,theta_z, h_l, zeta), bxinstance, "numpy")
byfunc = lambdify((h, phi, theta_x,theta_z, h_l, zeta), byinstance, "numpy")




mapping1 = computeAffineMapping(boundlines1,phi01,x01,y01,thetaX1,-0.0086,hval1,zetaval1,focalLengthPixels)
mapping2 = computeAffineMapping(boundlines2,phi02,x02,y02,thetaX2,-0.0086,hval2,zetaval2,focalLengthPixels)
mapping3 = computeAffineMapping(boundlines3,phi03,x03,y03,thetaX3,-0.0086,hval3,zetaval3,focalLengthPixels)


###
imgnew1 = Image.new( 'RGB', (bottleImage1.size[1],bottleImage1.size[0]), "black") 
imgnew2 = Image.new( 'RGB', (bottleImage2.size[1],bottleImage2.size[0]), "black") 
imgnew3 = Image.new( 'RGB', (bottleImage3.size[1],bottleImage3.size[0]), "black")


aspectRatio = 2.2    
print phi01, thetaX1, mapping1, hval1, zetaval1, aspectRatio
changeshapeInverseOptics(imgnew1,bottle1rotnd, phi01, thetaX1,thetaZ1, mapping1, hval1, zetaval1, aspectRatio)
imgrot1 = imgnew1.rotate(90)
im1 = np.asarray(imgrot1)
rightcrop1 = im1.shape[1]-60
bottomcrop1 = im1.shape[0]-60
imcropped1 = im1[60:bottomcrop1,60:rightcrop1]

print phi02, thetaX2, mapping2, hval1, hval2, aspectRatio

changeshapeInverseOptics(imgnew2,bottle2rotnd, phi02, thetaX2,thetaZ2, mapping2, hval2, zetaval2, aspectRatio)
imgrot2 = imgnew2.rotate(90)
im2 = np.asarray(imgrot2)
rightcrop2 = im2.shape[1]-80
bottomcrop2 = im2.shape[0]-60
imcropped2 = im2[60:bottomcrop2,100:rightcrop2]

print  phi03, thetaX3, mapping3, hval3, zetaval3, aspectRatio

changeshapeInverseOptics(imgnew3,bottle3rotnd, phi03, thetaX3,thetaZ3, mapping3, hval3, zetaval3, aspectRatio)
imgrot3 = imgnew3.rotate(90)
im3 = np.asarray(imgrot3)
rightcrop3 = im3.shape[1]-60
bottomcrop3 = im3.shape[0]-60
imcropped3 = im3[60:bottomcrop3,60:rightcrop3]

## STITCHING
img = Image.fromarray(imcropped1, 'RGB')
img.save('bot1.png')
img = Image.fromarray(imcropped2, 'RGB')
img.save('bot2.png')
img = Image.fromarray(imcropped3, 'RGB')
img.save('bot3.png')

'''
for i in range(numphotos):
    mapping = computeAffineMapping(boundlines[i],phi0[i],x0[i],y0[i],thetaX[i],-0.0086,hval[i],zetaval[i],focalLengthPixels)
    imgnew = Image.new( 'RGB', (bottleImage[i].size[1],bottleImage[i].size[0]), "black") 
    changeshapeInverseOptics(imgnew,bottlerotnd[i], phi0[i], thetaX[i], mapping, hval[i], zetaval[i], aspectRatio)
    imgrot = imgnew.rotate(90)
    im = np.asarray(imgrot)
    rightcrop = im.shape[1]-60
    bottomcrop = im[i].shape[0]-60
    imcropped = im[i][60:bottomcrop,60:rightcrop
    img = Image.fromarray(imcropped, 'RGB')
    img.save('bot' + i + '.png')
'''


inputdir = './input'
outputdir = './output'
keyframe_image = 'bot1.png'
AlignAndStitchImagesRansac((inputdir,keyframe_image,outputdir))


## OCR
import tesseract
import ctypes
import os


api = tesseract.TessBaseAPI()
api.SetOutputName("outputName");
api.Init(".","eng",tesseract.OEM_DEFAULT)
api.SetPageSegMode(tesseract.PSM_AUTO)
mImgFile = "./output/trialone.png"
api.SetRectangle(100,100,700,100)
result = tesseract.ProcessPagesFileStream(mImgFile,api)
print "result(ProcessPagesFileStream)=",result
