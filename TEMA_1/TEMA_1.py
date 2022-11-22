import cv2 as cv
import numpy as np
import os


def show_image(title, image):
    # image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def extrage_careu(image):
    image = cv.resize(image,(0,0),fx=0.2,fy=0.2)
    image = image[240:680,90:555]
    # print(image.shape)
    # show_image("imaged_croped",image)
    # low_yellow = (0, 0, 0)
    # high_yellow = (255, 120, 255)
    low_yellow = (0, 0, 0)
    high_yellow = (255, 140, 255)
 
    img_hsv = cv.cvtColor(image.copy(), cv.COLOR_BGR2HSV)
    mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
    # show_image("mask_yellow_hsv",mask_yellow_hsv)
    # image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    # image_m_blur = cv.medianBlur(img_hsv,3)
    # image_g_blur = cv.GaussianBlur(image_m_blur, (1, 1), 4) 
    # image_sharpened = cv.addWeighted(image_m_blur, 1.2, image_g_blur, -0.8, 0)
    # show_image('image_sharpened',image_sharpened) 
    # _, thresh = cv.threshold(image_sharpened, 30, 255, cv.THRESH_BINARY)

    # kernel = np.ones((1,1 ), np.uint8)
    # thresh = cv.erode(thresh, kernel)
    # show_image('image_thresholded',thresh)

    # edges =  cv.Canny(thresh ,0,0)
    # show_image('edges',edges)
    contours, _ = cv.findContours(mask_yellow_hsv,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    max_area = 0
   
    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            possible_bottom_left = None
            possible_top_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point
                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point
                
                
            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    width = 15*40
    height = 15*40
    
    image_copy = image.copy()
    cv.circle(image_copy,tuple(top_left),10,(0,0,255),-1) #red
    cv.circle(image_copy,tuple(top_right),10,(0,255,255),-1) #yellow
    cv.circle(image_copy,tuple(bottom_left),10,(255,0,255),-1)#pink
    cv.circle(image_copy,tuple(bottom_right),10,(255,255,255),-1)#white
    # show_image("detected corners",image_copy)

    puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")

    M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)

    result = cv.warpPerspective(image, M, (width, height))
    # result = cv.cvtColor(result,cv.COLOR_GRAY2BGR)
    
    return result

lines_horizontal=[]
for i in range(0,601,40):
    l=[]
    l.append((0,i))
    l.append((599,i))
    lines_horizontal.append(l)

lines_vertical=[]
for i in range(0,601,40):
    l=[]
    l.append((i,0))
    l.append((i,599))
    lines_vertical.append(l)

files=os.listdir("G:\GitHub Repositories\CAVA_Project_1\TEMA_1\\antrenare")
for file in files:
    break
    if file[-3:]=='jpg':
        img = cv.imread('G:\GitHub Repositories\CAVA_Project_1\TEMA_1\\antrenare\\'+file)
        result=extrage_careu(img)
        for line in  lines_vertical : 
            cv.line(result, line[0], line[1], (0, 255, 0), 5)
        for line in  lines_horizontal : 
            cv.line(result, line[0], line[1], (0, 0, 255), 5)
        show_image('img',result)

def determina_configuratie_careu_ox(img_hsv,lines_horizontal,lines_vertical):
    matrix = np.empty((15,15), dtype='str')
    # show_image("mask_hsv",img_hsv)
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0]+5
            y_max = lines_vertical[j + 1][1][0]-5
            x_min = lines_horizontal[i][0][1]+5
            x_max = lines_horizontal[i + 1][1][1]-5
            patch = img_hsv[x_min:x_max, y_min:y_max].copy()
            # show_image("mask_hsv",patch)
            Medie_patch=np.mean(patch)
            if Medie_patch>8:
                matrix[i][j]='x'
            else:
                matrix[i][j]='o'
    return matrix

def vizualizare_configuratie(result,matrix,lines_horizontal,lines_vertical):
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]
            if matrix[i][j] == 'x': 
                cv.rectangle(result, (y_min, x_min), (y_max, x_max), color=(255, 0, 0), thickness=5)

alfabet = ["A","B","C","D","E","F","G","H","I","J","L","M","N","O","P","R","S","T","U","V","X","Z","0"]
nr2 = len(alfabet)
for j in range(nr2):
    alfabet.append(alfabet[j]+"2")

def memoreaza_templates(img_hsv,img):
    index = 0
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            nr = 4
            y_min = lines_vertical[j][0][0]+nr
            y_max = lines_vertical[j + 1][1][0]-nr
            x_min = lines_horizontal[i][0][1]+nr
            x_max = lines_horizontal[i + 1][1][1]-nr
            patch = img_hsv[x_min:x_max, y_min:y_max].copy()
            patch_org = img[x_min:x_max, y_min:y_max].copy()
            # show_image("mask_hsv",patch)
            Medie_patch=np.mean(patch)
            if Medie_patch>8:
                filename = alfabet[index]+".jpg"
                index +=1
                cv.imwrite("G:\GitHub Repositories\CAVA_Project_1\TEMA_1\\templates\\"+filename,patch_org)
    return

files=os.listdir("G:\GitHub Repositories\CAVA_Project_1\TEMA_1\\antrenare")
for file in files:
    if file[-3:]=='jpg':
        break
        img = cv.imread("G:\GitHub Repositories\CAVA_Project_1\TEMA_1\\antrenare\\"+file)
        result=extrage_careu(img)
        low_yellow = (0, 0, 239)
        high_yellow = (255, 111, 255)
        img_hsv = cv.cvtColor(result.copy(), cv.COLOR_BGR2HSV)
        mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
        # res = cv.bitwise_and(img_hsv,img_hsv,mask = mask_yellow_hsv)
        # show_image("res",res)
        matrice=determina_configuratie_careu_ox(mask_yellow_hsv,lines_horizontal,lines_vertical)
        # print(matrice)
        vizualizare_configuratie(result,matrice,lines_horizontal,lines_vertical)
        show_image('img',result)
        break



# img = cv.imread("G:\GitHub Repositories\CAVA_Project_1\TEMA_1\imagini_auxiliare\litere_1.jpg")
# result=extrage_careu(img)
# low_yellow = (0, 0, 239)
# high_yellow = (255, 111, 255)
# img_hsv = cv.cvtColor(result.copy(), cv.COLOR_BGR2HSV)
# mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
# memoreaza_templates(mask_yellow_hsv,result)



def clasifica_litera(patch):
        maxi=-np.inf
        litera=""
        # show_image("patch",patch)
        patch = cv.cvtColor(patch.copy(),cv.COLOR_BGR2GRAY)
        for j in alfabet:
            img_template=cv.imread('G:\GitHub Repositories\CAVA_Project_1\TEMA_1\\templates\\'+j+'.jpg')
            img_template= cv.cvtColor(img_template,cv.COLOR_BGR2GRAY)
            # show_image("temp",img_template)
            corr = cv.matchTemplate(patch,img_template,  cv.TM_CCOEFF_NORMED)
            corr=np.max(corr)
            if corr>maxi :
                maxi=corr
                litera=j
        return litera

def determina_configuratie_careu_olitere(img_hsv,lines_horizontal,lines_vertical,img_original):
    matrix = np.empty((15,15), dtype='str')
    # show_image("mask_hsv",img_hsv)
    for i in range(len(lines_horizontal)-1):
        for j in range(len(lines_vertical)-1):
            y_min = lines_vertical[j][0][0]+5
            y_max = lines_vertical[j + 1][1][0]-5
            x_min = lines_horizontal[i][0][1]+5
            x_max = lines_horizontal[i + 1][1][1]-5
            patch = img_hsv[x_min:x_max, y_min:y_max].copy()
            y_min = lines_vertical[j][0][0]+8
            y_max = lines_vertical[j + 1][1][0]-8
            x_min = lines_horizontal[i][0][1]+8
            x_max = lines_horizontal[i + 1][1][1]-8
            patch_original = img_original[x_min:x_max, y_min:y_max].copy()
            # show_image("mask_hsv",patch)
            Medie_patch=np.mean(patch)
            if Medie_patch>8:
                matrix[i][j]=clasifica_litera(patch_original)
            else:
                matrix[i][j]="o"
    return matrix

matrice_viz= np.zeros((15,15), dtype='int')
files=os.listdir("G:\GitHub Repositories\CAVA_Project_1\TEMA_1\\antrenare")
for file in files:
    if file[-3:]=='jpg':
        img = cv.imread("G:\GitHub Repositories\CAVA_Project_1\TEMA_1\\antrenare\\"+file)
        result=extrage_careu(img)
        low_yellow = (0, 0, 239)
        high_yellow = (255, 111, 255)
        img_hsv = cv.cvtColor(result.copy(), cv.COLOR_BGR2HSV)
        mask_yellow_hsv = cv.inRange(img_hsv, low_yellow, high_yellow)
        matrice=determina_configuratie_careu_olitere(mask_yellow_hsv,lines_horizontal,lines_vertical,result)
        # print(matrice)
        # show_image('img',result)



