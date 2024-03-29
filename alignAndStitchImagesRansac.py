
import os
import sys
import cv2
import math
import numpy as np
import utils

from numpy import linalg
from sys import argv,exit



class AlignAndStitchImagesRansac(object):
    def __init__(self, sysArgs):
    #def __init__(self, image_dir, key_frame, output_dir, img_filter=None):
        '''
        image_dir: 'directory' containing all images
        key_frame: 'dir/name.jpg' of the base image
        output_dir: 'directory' where to save output images
        optional: 
           img_filter = 'JPG'; None->Take all images
        '''
        img_filter = None
        path = os.getcwd()
        print os.getcwd()
        image_dir = os.path.join(path,sysArgs[0])
        key_frame = os.path.join(image_dir,sysArgs[1])
        output_dir = os.path.join(path,sysArgs[2])
        self.key_frame_file = os.path.split(key_frame)[-1]
        self.output_dir = output_dir
        
        print "key_frame",key_frame
        # Open the directory given in the arguments
        self.dir_list = []
        try:
            self.dir_list = os.listdir(image_dir)
            if img_filter:
                # remove all files that doen't end with .[image_filter]
                self.dir_list = filter(lambda x: x.find(img_filter) > -1, self.dir_list)
            try: #remove Thumbs.db, is existent (windows only)
                self.dir_list.remove('Thumbs.db')
            except ValueError:
                pass
                
        
        except:
            print >> sys.stderr, ("Unable to open directory: %s" % image_dir)
            sys.exit(-1)
    
        self.dir_list = map(lambda x: os.path.join(image_dir, x), self.dir_list)
        
        self.dir_list = filter(lambda x: x != key_frame, self.dir_list)
    
        
        base_img_rgb = cv2.imread(key_frame)
        if base_img_rgb == None:
            raise IOError("%s doesn't exist" %key_frame)
        # utils.showImage(base_img_rgb, scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()
        
        final_img = self.stitchImages(base_img_rgb, 0)        
        


    def filter_matches(self, matches, ratio = 0.75):
        filtered_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                filtered_matches.append(m[0])
        
        return filtered_matches
    
    def imageDistance(self, matches):
    
        sumDistance = 0.0
    
        for match in matches:
    
            sumDistance += match.distance
    
        return sumDistance
    
    def findDimensions(self, image, homography):
        base_p1 = np.ones(3, np.float32)
        base_p2 = np.ones(3, np.float32)
        base_p3 = np.ones(3, np.float32)
        base_p4 = np.ones(3, np.float32)
    
        (y, x) = image.shape[:2]
    
        base_p1[:2] = [0,0]
        base_p2[:2] = [x,0]
        base_p3[:2] = [0,y]
        base_p4[:2] = [x,y]
    
        max_x = None
        max_y = None
        min_x = None
        min_y = None
    
        for pt in [base_p1, base_p2, base_p3, base_p4]:
    
            hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
    
            hp_arr = np.array(hp, np.float32)
    
            normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)
    
            if ( max_x == None or normal_pt[0,0] > max_x ):
                max_x = normal_pt[0,0]
    
            if ( max_y == None or normal_pt[1,0] > max_y ):
                max_y = normal_pt[1,0]
    
            if ( min_x == None or normal_pt[0,0] < min_x ):
                min_x = normal_pt[0,0]
    
            if ( min_y == None or normal_pt[1,0] < min_y ):
                min_y = normal_pt[1,0]
    
        min_x = min(0, min_x)
        min_y = min(0, min_y)
    
        return (min_x, min_y, max_x, max_y)

    
    def stitchImages(self, base_img_rgb, round=0):
    
        if ( len(self.dir_list) < 1 ):
            return base_img_rgb
    
       # print base_img_rgb.channels()
    # if(image.channels()==1)
    # {      /* Grayscale */                                             }
    # else if (image.channels==4)
    # {      /* ARGB or RGBA image */    
    
    
        base_img = cv2.GaussianBlur(cv2.cvtColor(base_img_rgb,cv2.COLOR_BGR2GRAY), (5,5), 0)
    
        # Use the SIFT feature detector
        detector = cv2.SIFT()
    
        # Find key points in base image for motion estimation
        base_features, base_descs = detector.detectAndCompute(base_img, None)
    
        # Create new key point list
        # key_points = []
        # for kp in base_features:
        #     key_points.append((int(kp.pt[0]),int(kp.pt[1])))
        # utils.showImage(base_img, key_points, scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()
    
        # Parameters for nearest-neighbor matching
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE, 
            trees = 5)
        matcher = cv2.FlannBasedMatcher(flann_params, {})
    
        print "Iterating through next images..."
    
        closestImage = None
    
        # TODO: Thread this loop since each iteration is independent
    
        # Find the best next image from the remaining images
        for next_img_path in self.dir_list:
    
            print "Reading %s..." % next_img_path
    
            if ( self.key_frame_file in next_img_path ):
                print "\t Skipping %s..." % self.key_frame_file
                continue
    
            # Read in the next image...
            next_img_rgb = cv2.imread(next_img_path)
            next_img = cv2.GaussianBlur(cv2.cvtColor(next_img_rgb,cv2.COLOR_BGR2GRAY), (5,5), 0)
    
            # if ( next_img.shape != base_img.shape ):
            #     print "\t Skipping %s, bad shape: %s" % (next_img_path, next_img.shape)
            #     continue
    
            print "\t Finding points..."
    
            # Find points in the next frame
            next_features, next_descs = detector.detectAndCompute(next_img, None)
    
            matches = matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)
    
            print "\t Match Count: ", len(matches)
    
            matches_subset = self.filter_matches(matches)
    
            print "\t Filtered Match Count: ", len(matches_subset)
    
            distance = self.imageDistance(matches_subset)
    
            print "\t Distance from Key Image: ", distance
    
            averagePointDistance = distance/float(len(matches_subset))
    
            print "\t Average Distance: ", averagePointDistance
    
            kp1 = []
            kp2 = []
    
            for match in matches_subset:
                kp1.append(base_features[match.trainIdx])
                kp2.append(next_features[match.queryIdx])
    
            p1 = np.array([k.pt for k in kp1])
            p2 = np.array([k.pt for k in kp2])
    
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            print '%d / %d  inliers/matched' % (np.sum(status), len(status))
    
            inlierRatio = float(np.sum(status)) / float(len(status))
            
            # if ( closestImage == None or averagePointDistance < closestImage['dist'] ):
            if ( closestImage == None or inlierRatio > closestImage['inliers'] ):
                closestImage = {}
                closestImage['h'] = H
                closestImage['inliers'] = inlierRatio
                closestImage['dist'] = averagePointDistance
                closestImage['path'] = next_img_path
                closestImage['rgb'] = next_img_rgb
                closestImage['img'] = next_img
                closestImage['feat'] = next_features
                closestImage['desc'] = next_descs
                closestImage['match'] = matches_subset
    
        print "Closest Image: ", closestImage['path']
        print "Closest Image Ratio: ", closestImage['inliers']

        self.dir_list = filter(lambda x: x != closestImage['path'], self.dir_list)
    
        # utils.showImage(closestImage['img'], scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()
    
        H = closestImage['h']
        H = H / H[2,2]
        H_inv = linalg.inv(H)
    
        if ( closestImage['inliers'] > 0.1 ): # and 
    
            (min_x, min_y, max_x, max_y) = self.findDimensions(closestImage['img'], H_inv)
    
            # Adjust max_x and max_y by base img size
            max_x = max(max_x, base_img.shape[1])
            max_y = max(max_y, base_img.shape[0])
    
            move_h = np.matrix(np.identity(3), np.float32)
    
            if ( min_x < 0 ):
                move_h[0,2] += -min_x
                max_x += -min_x
    
            if ( min_y < 0 ):
                move_h[1,2] += -min_y
                max_y += -min_y
    
            print "Homography: \n", H
            print "Inverse Homography: \n", H_inv
            print "Min Points: ", (min_x, min_y)
    
            mod_inv_h = move_h * H_inv
    
            img_w = int(math.ceil(max_x))
            img_h = int(math.ceil(max_y))
    
            print "New Dimensions: ", (img_w, img_h)
    
            # Warp the new image given the homography from the old image
            base_img_warp = cv2.warpPerspective(base_img_rgb, move_h, (img_w, img_h))
            print "Warped base image"
    
            # utils.showImage(base_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()
    
            next_img_warp = cv2.warpPerspective(closestImage['rgb'], mod_inv_h, (img_w, img_h))
            print "Warped next image"
    
            # utils.showImage(next_img_warp, scale=(0.2, 0.2), timeout=5000)
            # cv2.destroyAllWindows()
    
            # Put the base image on an enlarged palette
            enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)
    
            print "Enlarged Image Shape: ", enlarged_base_img.shape
            print "Base Image Shape: ", base_img_rgb.shape
            print "Base Image Warp Shape: ", base_img_warp.shape
    
            # enlarged_base_img[y:y+base_img_rgb.shape[0],x:x+base_img_rgb.shape[1]] = base_img_rgb
            # enlarged_base_img[:base_img_warp.shape[0],:base_img_warp.shape[1]] = base_img_warp
    
            # Create a mask from the warped image for constructing masked composite
            (ret,data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY), 
                0, 255, cv2.THRESH_BINARY)
    
            enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, 
                mask=np.bitwise_not(data_map), 
                dtype=cv2.CV_8U)
    
            # Now add the warped image
            final_img = cv2.add(enlarged_base_img, next_img_warp, 
                dtype=cv2.CV_8U)
    
            # utils.showImage(final_img, scale=(0.2, 0.2), timeout=0)
            # cv2.destroyAllWindows()
    
            # Crop off the black edges
            final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            print "Found %d contours..." % (len(contours))
    
            max_area = 0
            best_rect = (0,0,0,0)
    
            for cnt in contours:
                x,y,w,h = cv2.boundingRect(cnt)
                # print "Bounding Rectangle: ", (x,y,w,h)
    
                deltaHeight = h-y
                deltaWidth = w-x
    
                area = deltaHeight * deltaWidth
    
                if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
                    max_area = area
                    best_rect = (x,y,w,h)
    
            if ( max_area > 0 ):
                print "Maximum Contour: ", max_area
                print "Best Rectangle: ", best_rect
    
                final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],
                        best_rect[0]:best_rect[0]+best_rect[2]]
    
                # utils.showImage(final_img_crop, scale=(0.2, 0.2), timeout=0)
                # cv2.destroyAllWindows()
    
                final_img = final_img_crop
    
            # Write out the current round
            final_filename = "%s/%d.JPG" % (self.output_dir, round)
            cv2.imwrite(final_filename, final_img)
    
            return self.stitchImages(final_img, round+1)
    
        else:
    
            return self.stitchImages(base_img_rgb, round+1)
    
    