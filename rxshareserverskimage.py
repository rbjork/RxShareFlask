from __future__ import division
from sympy import symbols,cos,sin,tan,pi
from sympy.utilities.lambdify import lambdify
from PIL import Image
import Image as Img
import numpy as np
from scipy import ndimage
#import cv2
import math
from subprocess import call
import os
from alignAndStitchImagesRansac import AlignAndStitchImagesRansac
from flask import render_template
from flask import Flask, request, redirect, url_for
from werkzeug import secure_filename
from skimage.transform import (hough_line, hough_line_peaks)
from skimage import filter
from skimage import data
import random


## CELL
def getMidline(params,bottle):  # Returns slope intercept coefficients of bounding lines and midline
    height = bottle.shape[0]
    #width = bottle.shape[1]
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
    
    #if testing:
       # cv2.line(bottle,(x1,y1),(x2,y2),(0,0,255),2)
    
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
    
    #if testing:
        #cv2.line(bottle,(x1,y1),(x2,y2),(0,0,255),2)
    
    mx2 = (x2 - x1)/float(y2 - y1)
    bx2 = x2 - mx2*y2
    
    mx = (mx1 + mx2)/2  # slope between
    bx = (bx1 + bx2)/2  # intercept with x axis
    xm = mx*height + bx
    
#    print "point slope ",bx1,mx1,bx2,mx2
    #if testing:
        #cv2.line(bottle,(int(bx),0),(int(xm),height),(255,0,255),2) 
    
    return (mx1,bx1, mx2, bx2, mx, bx)
    
# FILTER LEFT AND RIGHT

def getKey(item):
    return item[0]


def houghSides(bottle,edges, threshold, left):
    h, theta, d = hough_line(edges)
    accum = zip(*hough_line_peaks(h, theta, d))
    sortedAccum = sorted(accum,key = getKey,reverse=True) 
    sortedAccumR = [sa for sa in sortedAccum if sa[2] > 200]
    sortedAccumL = [sa for sa in sortedAccum if sa[2] <= 200]
    
    
    if left:
        hpeak, angle, dist = sortedAccumL[0]
    else:
        hpeak, angle, dist in sortedAccumR[0]

    return (1,dist,angle)       
    #print "LEFT 0",sortedAccumL[0]    
    
                       
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
    #if testing:
        #cv2.circle(bottle, (x0,y0), 10, (0,127,0),8)
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
    xy_multiplier = 1.02*(pxchord/2)/ww    # This is chord at alpha0  (Not quite chord Z but hopefully close enough)
    xzero = -xy_multiplier*bxfunc(0,math.pi/2,thetaX,thetaZ,hlen,zeta)
    xoffset = x0 - xzero - 2
    
    #yzero = -xy_multiplier*byfunc(0,math.pi/2,thetaX,hlen,zeta)
    yzero = -.95*xy_multiplier*byfunc(0,math.pi/2,thetaX,thetaZ,hlen,zeta)
    
    yoffset = y0 - yzero
    print yzero
    return (xy_multiplier,xoffset,yoffset)
    
def displayMappings(bottle, phi0, thetaX,thetaZ, beta, hLenVal, mapping, boundlines):
    
    xy_multiplier = mapping[0]
    xoffset = mapping[1]
    yoffset = mapping[2]
    
    #endAngle = math.pi - alpha0
    angles = np.linspace(phi0, math.pi-phi0 ,50)
    
    ####  NOTE THE FUDGE FACTOR ADDED !!! ####
    ## (bx,by) = (1.09,-.95), (1.07,-.95),(1.05,-.95)
    
    bxvals = xy_multiplier*bxfunc(hLenVal,angles,thetaX,thetaZ,hLenVal,beta) + xoffset  
    byvals = -xy_multiplier*byfunc(hLenVal,angles,thetaX,thetaZ,hLenVal,beta) + yoffset
    
    #ax.plot(bxvals , byvals,  lw=1, c="orange") 
    if testing:
        x1 = int(bxvals[0])
        y1 = int(byvals[0])
        for i in range(len(bxvals)-1):
            x2 = int(bxvals[i+1])
            y2 = int(byvals[i+1])
            #cv2.line(bottle,(x1,y1),(x2,y2),(0,0,255),2)
            x1 = x2
            y1 = y2
    
    bxvals = xy_multiplier*bxfunc(4,angles,thetaX,thetaZ,hLenVal,beta) + xoffset 
    byvals = -xy_multiplier*byfunc(4,angles,thetaX,thetaZ,hLenVal,beta) + yoffset
    
    if testing:
        x1 = int(bxvals[0])
        y1 = int(byvals[0])
        for i in range(len(bxvals)-1):
            x2 = int(bxvals[i+1])
            y2 = int(byvals[i+1])
            #cv2.line(bottle,(x1,y1),(x2,y2),(0,255,255),2)
            x1 = x2
            y1 = y2
    min1 = np.min(bxvals)
    max1 = np.max(bxvals)
    width1 = (np.max(bxvals) - np.min(bxvals))
    
 
    bxvals = xy_multiplier*bxfunc(2,angles,thetaX,thetaZ,hLenVal,beta) + xoffset 
    byvals = -xy_multiplier*byfunc(2,angles,thetaX,thetaZ,hLenVal,beta) + yoffset
    
    if testing:
        x1 = int(bxvals[0])
        y1 = int(byvals[0])
        for i in range(len(bxvals)-1):
            x2 = int(bxvals[i+1])
            y2 = int(byvals[i+1])
            #cv2.line(bottle,(x1,y1),(x2,y2),(255,0,0),2)
            x1 = x2
            y1 = y2
    min2 = np.min(bxvals)
    max2 = np.max(bxvals)
    width2 = (np.max(bxvals) - np.min(bxvals))
   
    bxvals = xy_multiplier*bxfunc(0,angles,thetaX,thetaZ,hLenVal,beta) + xoffset 
    byvals = -xy_multiplier*byfunc(0,angles,thetaX,thetaZ,hLenVal,beta) + yoffset
    
    if testing:
        x1 = int(bxvals[0])
        y1 = int(byvals[0])
        for i in range(len(bxvals)-1):
            x2 = int(bxvals[i+1])
            y2 = int(byvals[i+1])
            #cv2.line(bottle,(x1,y1),(x2,y2),(255,0,255),2)
            x1 = x2
            y1 = y2
            
    min3 = np.min(bxvals)
    max3 = np.max(bxvals)
    width3 = (np.max(bxvals) - np.min(bxvals))
    print "width",width1,width2,width3
    print "min max",(min1,max1),(min2,max2),(min3,max3)


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
    
    
def loadimages():
    return uploaded_files #['bot1stripedH.jpg','bot2stripedH.jpg','bot3stripedH.jpg']

def edges():
    pass

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def process():
    global focalLengthPixels
    global sensorRes
    global aspectRatio
    global FOVw
    global FOVh
    global fov
    leftcrp = (120,200,200)
    rightcrp = (200,200,120)
    images = loadimages()
    nameIndex = 1
    paramIndex = 0
    for img in images:
        bottleImage1 = Image.open('./uploads/' + img)
        bottle1rotnd = np.asarray(bottleImage1)
        
        img1 = ndimage.gaussian_filter(bottle1rotnd, 1)
        
        #gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray1 = rgb2gray(img1)
        edges1 = filter.canny(gray1,sigma=5)
        #edges1 = cv2.Canny(gray1,20,150, apertureSize = 3, L2gradient = True)
        
        params1 = houghManage(bottle1rotnd,edges1,100)
        boundlines1 = getMidline(params1,bottle1rotnd)
        x01, y01 = findMidlineEllipseIntercept(edges1,boundlines1,bottle1rotnd)

        sensorRes = bottle1rotnd.shape  # if reduced
        focalLengthPixels = sensorRes[0]/2*cot(FOVw/2*math.pi/180)
        fov = getFOV(sensorRes,focalLengthPixels)
        
        thetaX1 = computeTheta(boundlines1,focalLengthPixels,sensorRes)
        beta1 = computeBeta(x01,y01,sensorRes,focalLengthPixels)
        phi01 = computePhi0(boundlines1,sensorRes,thetaX1,focalLengthPixels)
        phi01 = abs(phi01)
        hval1 = computeHlensRadRatio(beta1,phi01,thetaX1)
        zetaval1 = computeZeta(beta1, phi01, thetaX1)
        #Z1 = (thetaX1 + math.pi/2) - zetaval1
        thetaZ1 = computeThetaZ(boundlines1)

        mapping1 = computeAffineMapping(boundlines1,phi01,x01,y01,thetaX1,-0.0086,hval1,zetaval1,focalLengthPixels)
        
        if testing:
            displayMappings(bottle1rotnd, phi01, thetaX1,thetaZ1, beta1, hval1, mapping1, boundlines1)
            img = Image.fromarray(bottle1rotnd, 'RGB')
            img.save('test' + str(nameIndex) + '.png')
            bottle1rotnd = np.asarray(bottleImage1)
        
        imgnew1 = Image.new( 'RGB', (bottleImage1.size[1],bottleImage1.size[0]), "black")
           
        print phi01, thetaX1, mapping1, hval1, zetaval1, aspectRatio
        changeshapeInverseOptics(imgnew1,bottle1rotnd, phi01, thetaX1,thetaZ1, mapping1, hval1, zetaval1, aspectRatio)
        imgrot1 = imgnew1.rotate(90)
        im1 = np.asarray(imgrot1)
        rightcrop1 = im1.shape[1]-rightcrp[paramIndex]
        bottomcrop1 = im1.shape[0]-60
        imcropped1 = im1[10:bottomcrop1,leftcrp[paramIndex]:rightcrop1]
        img = Image.fromarray(imcropped1, 'RGB')
        img.save('./input/bot' + str(nameIndex) + '.png')
        nameIndex = nameIndex + 1
        paramIndex = paramIndex + 1
        
        
    AlignAndStitchImagesRansac((inputdir,keyframe_image,outputdir))
    '''
    api = tesseract.TessBaseAPI()
    api.SetOutputName("outputName");
    api.Init(".","eng",tesseract.OEM_DEFAULT)
    api.SetPageSegMode(tesseract.PSM_AUTO)
    mImgFile = "./output/trialone.png"
    api.SetRectangle(100,100,700,100)
    result = tesseract.ProcessPagesFileStream(mImgFile,api)
    print "result(ProcessPagesFileStream)=",result 
    '''


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 6 * 1224 * 1632

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/processsubmitted', methods=['GET', 'POST'])
def upload_file():
    print "upload method called"
    if request.method == 'POST':
        print "upload method POST"
        file = request.files['file']
        if file and allowed_file(file.filename):
            print "upload method",file.filename
            filename = file.filename #secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('processlocalimages',filenames=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/upload',methods=['GET','POST'])
def upload():
    global uploaded_files
    print "uploadphotos"
    if request.method == 'POST':
        print "uploadphotos with POST"
        uploaded_files = request.files.getlist("file[]")
        filenames = []
        for file in uploaded_files:
            # Check if the file is one of the allowed types/extensions
            if file and allowed_file(file.filename):
                # Make the filename safe, remove unsupported chars
                filename = secure_filename(file.filename)
                print filename
                # Move the file form the temporal folder to the upload
                # folder we setup
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # Save the filename into a list, we'll use it later
                filenames.append(filename)
                # Redirect the user to the uploaded_file route, which
                # will basicaly show on the browser the uploaded file
                # Load an html page with a link to each uploaded file
        print "all files",filenames
        return redirect(url_for('processlocalimages',filenames=filenames))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="upload" method=post enctype=multipart/form-data>
      <input type="file" multiple="" name="file[]"  /><br />
      <input type=submit value=Upload>
    </form>
    '''
    
def valid_login(un,pw):
    return True
    
def log_the_user_in(un):
    pass

@app.route('/login', methods=['POST', 'GET'])
def login():
    error = None
    if request.method == 'POST':
        if valid_login(request.form['username'],
                       request.form['password']):
            return log_the_user_in(request.form['username'])
        else:
            error = 'Invalid username/password'
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('login.html', error=error)
    
@app.route("/processlocal/<filenames>")
def processlocalimages(filenames=None):
    print "processlocalimages method called"
    process()
    
    call("Tesseract output/1.JPG ocrout")
    return render_template('completed.html',filename=filenames, imgout="./output/1.JPG")

@app.route("/")
@app.route("/<name>")
def home(name=None):
    return render_template('welcome.html',name=name)
    
ocr = ["Tesseract"]

if __name__ == "__main__":
    inputdir = './input'
    outputdir = './output'
    keyframe_image = 'bot1.png'
    rval = 1
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
    
    bxinstance = bx.subs([(r,rval)])
    byinstance = by.subs([(r,rval)])
    
    bxfunc = lambdify((h, phi, theta_x,theta_z, h_l, zeta), bxinstance, "numpy")
    byfunc = lambdify((h, phi, theta_x,theta_z, h_l, zeta), byinstance, "numpy")
    
    sensorSize = (4.6, 3.5) # mm
    
    focalLengthPixels = 0
    sensorRes = (0,0)
    aspectRatio = 2.2 
    FOVw, FOVh = (60,48) # units of degrees
    fov = (60,40)
    testing = True
    
    pid = os.getpid()
    
    uploaded_files = []
    
    app.run()


    



   
   





   


    

    