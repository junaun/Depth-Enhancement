import numpy as np
import cv2
import matplotlib.pyplot as plt

# img = cv2.imread("C:/Users/ZHY/Downloads/Data-20230308T102753Z-001/Data/Mechmind/070323_1804H/color/01.png")
# depth = cv2.imread("C:/Users/ZHY/Downloads/Data-20230308T102753Z-001/Data/Mechmind/070323_1804H/depth/01.png0", cv2.IMREAD_ANYDEPTH).astype(float)
#
# img = cv2.resize(img, (640, 512))
# depth = cv2.resize(depth, (640, 512))
num = "000"
folder = "030523_1120H"
# fitness = np.load(f"combined_data/{folder}/fitness.npy")
# inlier_rmse = np.load(f"combined_data/{folder}/inlier_rmse.npy")
x_pos = 300
y_pos = 150

def normalize(depth):
    depth = (depth-400)/1500*255
    depth = depth.astype(np.uint8)
    depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
    return depth

for num in range(256):
    img_r = cv2.imread(f"combined_data/{folder}/color/{num:03}_real.png")
    depth_r = cv2.imread(f"combined_data/{folder}/depth/{num:03}_real.png", cv2.IMREAD_ANYDEPTH).astype(float)

    # depth_r1d = depth_r.reshape(-1)
    # print(depth_r1d.shape)
    #
    # # Plot histogram
    # plt.hist(depth_r1d, bins=1000)
    #
    # # Add labels and title
    # plt.xlabel('Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram')
    #
    # # Show plot
    # plt.show()

    img_m = cv2.imread(f"combined_data/{folder}/color/{num:03}_mech.png")
    depth_m = cv2.imread(f"combined_data/{folder}/depth/{num:03}_mech.png", cv2.IMREAD_ANYDEPTH).astype(float)

    img_r_crop = img_r[150:550,300:900]
    img_m_crop = img_m[150:550,300:900]
    depth_r_crop = normalize(depth_r[150:550,300:900])
    depth_m_crop = normalize(depth_m[150:550,300:900])

    img_rm_crop1 = np.hstack((img_r_crop, img_m_crop))
    img_rm_crop2 = np.hstack((depth_r_crop, depth_m_crop))
    img_rm_crop = np.vstack((img_rm_crop1, img_rm_crop2))
    cv2.imwrite("temp.png", img_rm_crop)


    def on_key_press(key):
        global x_pos, y_pos, img
        img = cv2.imread("temp.png")
        if key == 65361:  # left arrow key
            x_pos -= 1
            draw_line(x_pos, y_pos)  # shift line left
        elif key == 65363:  # right arrow key
            x_pos += 1
            draw_line(x_pos, y_pos)  # shift line right
        elif key == 65362:  # up arrow key
            y_pos -= 1
            draw_line(x_pos, y_pos)  # shift line up
        elif key == 65364:  # down arrow key
            y_pos += 1
            draw_line(x_pos, y_pos)  # shift line down

        cv2.imshow( f"Image",img)  # show updated image

    def draw_line(x, y):
        global img
        img[:,x,:] = 0
        img[:,x,2] = 255
        img[:,x+img_r_crop.shape[1],:] = 0
        img[:,x+img_r_crop.shape[1],2] = 255

        img[y,:,:] = 0
        img[y,:,2] = 255
        img[y+img_r_crop.shape[0],:,:] = 0
        img[y+img_r_crop.shape[0],:,2] = 255

        # img[:,310,:] = 0
        # img[:,310,1] = 255
        # img[:,310+img_r_crop.shape[1],:] = 0
        # img[:,310+img_r_crop.shape[1],1] = 255

    cv2.namedWindow("Image")
    cv2.imshow(f"Image", img_rm_crop)

    while True:
        key = cv2.waitKeyEx(0)
        if key == 113:
            cv2.destroyAllWindows()
            break
        else:
            on_key_press(key)

