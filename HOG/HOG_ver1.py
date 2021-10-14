import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
def get_differential_filter():
    
    filter_x = np.ones((3,3))
    filter_y = np.ones((3,3))
    
    filter_x[:,0] = -filter_x[:,0]
    filter_x[:,1] = 0
    
    filter_y[0,:] = -filter_y[:,0]
    filter_y[1,:] = 0
    return filter_x, filter_y


def convolution(im, kernel): #custon convolution function
    h,w = im.shape
    im_pad = np.pad(im, [(2,2),(2,2)], mode='constant')
    im_conv = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            cell = np.sum(kernel * im_pad[i:i+kernel.shape[0], j:j+kernel.shape[0]]) #assume square matrix kernel
            im_conv[i,j] = (cell)
    return im_conv

def filter_image(im, filters):
    
    gauss = np.ones((3,3)) 
    gauss = gauss/4
    gauss[1,1] = gauss[1,1] * 4
    gauss[0,1] = gauss[0,1] * 2
    gauss[1,0] = gauss[0,1] * 2
    gauss[2,1] = gauss[0,1] * 2
    gauss[1,2] = gauss[0,1] * 2 
    #print(gauss)
    
    im_blur = convolution(im, gauss) #blurring before differentiation with sample gaussian kernel
    im_filtered = convolution(im_blur, filters) #applying differentiation
    
    return im_filtered
'''
#testing parts of code
plt.figure()
a = plt.imread('einstein.jpg')
#a = np.zeros((10,10))
#a[:,5:] = np.ones((10,5))
plt.title("Before and after differentiation + blurring")
plt.subplot(1,3,1)
plt.title("Before")
plt.imshow(a[:,:,1])
plt.subplot(1,3,2)
plt.title("After: dy/dx")
a_filtx = filter_image(a[:,:,0],get_differential_filter()[0])
plt.imshow(a_filtx)
plt.subplot(1,3,3)
plt.title("After: dy/dx")
a_filty = filter_image(a[:,:,0],get_differential_filter()[1])
plt.imshow(a_filty)
plt.tight_layout()
plt.show()
'''
def get_gradient(im_dx, im_dy):
    tol=1e-5 #to prevent divide by 0 errors
    grad_mag = np.sqrt(im_dx**2 + im_dy**2)
    grad_angle = np.zeros((im_dx.shape[0],im_dx.shape[1]))
    for i in range(im_dx.shape[0]): #have to iterate through each element to catch negative angles
        for j in range(im_dx.shape[1]):
            if np.arctan(im_dy[i,j]/(im_dx[i,j]+tol)) <= 0: #added tolerance to prevent divide by zero errors
                grad_angle[i,j] = np.arctan(im_dy[i,j]/(im_dx[i,j]+tol))*(180/np.pi) + 180 #negative angles must be shifted by pi
            
            elif np.isclose(np.arctan(im_dy[i,j]/(im_dx[i,j]+tol)),np.pi,rtol=1e-15):
                grad_angle[i,j] = np.arctan(im_dy[i,j]/(im_dx[i,j]+tol))*(180/np.pi) - 180
            
            else: 
                grad_angle[i,j] = np.arctan(im_dy[i,j]/(im_dx[i,j]+tol))*(180/np.pi) #converted angles to degs from rads
    return grad_mag, grad_angle
#mag, ang = get_gradient(a_filtx,a_filty)


def build_histogram(grad_mag, grad_angle, cell_size):
    m,n=grad_mag.shape
    #print("m,n:",m,n)
    M = int(np.floor(m/cell_size))
    N = int(np.floor(n/cell_size))
    #print("M,N:",M,N)
    ori_histo = np.zeros((M,N,6))
    bins = np.zeros(6)
    #print(ori_histo.shape)
    for i in range(M):
        for j in range(N):
            for u in range(i*cell_size,i*cell_size + cell_size): #accessing the elements within a cell
                for v in range(j*cell_size,j*cell_size + cell_size):
                    bin_num = int(((grad_angle[u,v] + 15)%180)//30) #what bin a given angle should fall into)
                    ori_histo[i,j,bin_num] += grad_mag[u,v]
                    
                    
    return ori_histo
#ori_histo = build_histogram(mag, ang, 8)

def get_block_descriptor(ori_histo, block_size):
    tol = 1e-5 #to prevent divide by zero errors
    M = ori_histo.shape[0]
    N = ori_histo.shape[1]
    ori_histo_normalized = np.zeros((M - (block_size - 1),N - (block_size - 1), 6*block_size**2))
    #print(ori_histo_normalized.shape)
    for i in range(M - (block_size - 1)):
        for j in range(N - (block_size - 1)):
            #get the 2x2 (sample block size) vectors of HOGS, concatenate then normalize
            shift = 0
            for u in range(block_size):     #i*block_size,i*block_size+block_size
                for v in range(block_size):    #j*block_size,j*block_size+block_size
                    #print(i,j,u,v)
                    h_i = ori_histo[i+u,j+v,:] #need to simply shift (i,j)th block by u and v to get inner histograms                  
                    ori_histo_normalized[i,j,6*shift:6*shift+6] = h_i / np.sqrt(np.sum(h_i**2) + tol**2) #normalizing
                    shift+=1  #need to shift the index of normalized hist to fit 6 numbers every block_size times; cannot simply append-it's an array
                    #ori_histo_normalized[i,j] = np.concatenate(ori_histo[u,v], a2, ...))
                    
    return ori_histo_normalized



def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    filter_x = get_differential_filter()[0] #x differential filter
    filter_y = get_differential_filter()[1] #y differential filter
    
    dim_dx = filter_image(im, filter_x) #x differential
    dim_dy = filter_image(im, filter_y) #y differential
    
    grad_mag, grad_angle = get_gradient(dim_dx, dim_dy)
    
    cell_size = 8
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)
    
    block_size = 2
    ori_histo_normalized = get_block_descriptor(ori_histo, block_size)
    
    hog = np.ravel(ori_histo_normalized)

    # visualize to verify
    visualize_hog(im, hog, 8, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()





def face_recognition(I_target, I_template):

    #target_hog = extract_hog(I_target)
    template_hog = extract_hog(I_template)
    b = template_hog - np.mean(template_hog)
    bounding_boxes = []
    for i in (range(I_target.shape[0] - I_template.shape[0] +1)):
        for j in range(I_target.shape[1] - I_template.shape[1] +1):
            patch = I_target[i:i+I_template.shape[0], j:j+I_template.shape[1]] #extracting a 50x50 patch window
            patch_hog = extract_hog(patch) #getting the HOG of that window
            a = patch_hog - np.mean(patch_hog) #subtracting means
            
            s = np.correlate(a,b) #comparing the patch hog with the template hog
            if s > 0.5: #if they correlate more than 50% we append
                bounding_boxes.append([i,j,s])
    bounding_boxes = np.array(bounding_boxes)
    
    for i in range(len(bounding_boxes)):
        for j in range(i,len(bounding_boxes)):
            xi = bounding_boxes[i][0] #coordinates and correlation of ith bounding box
            yi = bounding_boxes[i][1]
            si = bounding_boxes[i][2]
            
            xj = bounding_boxes[j][0] #coordinates and correlation of subsequent bounding boxes
            yj = bounding_boxes[j][1]
            sj = bounding_boxes[j][2]
            area_overlap = np.abs(((xi + 50)*(yi+50)) - ((xj - xi) * (yj - yi))) #area of overlap
            percent_overlap = area_overlap/ (area_overlap + (xj+50)*(yj+50) + (xi+50)*(yi+50)) #percent of overlap
            if percent_overlap > 0.5:
                if si > sj:
                    del bounding_boxes[j] #deleting the smaller correlated object
                else:
                    del bounding_boxes[i]
    
   
    return  bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()




if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)

    I_target= cv2.imread('target.png', 0)
    #MxN image

    I_template = cv2.imread('template.png', 0)
    #mxn  face template
    print(I_target.shape, I_template.shape)
    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    




