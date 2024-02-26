# Import thư viện
import cv2
import mediapipe as mp
import imageio
import numpy as np

from google.protobuf.json_format import MessageToDict

#Set Camera
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

#Các biến cần dùng
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2)
mpDraw = mp.solutions.drawing_utils
i_punch = 0
i_ball = 0
i_thunder = 0
energy = 0
attack = False

#Effect cần dùng
thunderball = imageio.mimread('god_of_thunder/thunderball.gif')
thunder = imageio.mimread('god_of_thunder/thunder.gif')
thunder_punch = imageio.mimread('god_of_thunder/thunder_first.gif')

#Xử lý effect
nums_thunderball = len(thunderball)
thunderball_img = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in thunderball]
nums_thunder = len(thunder)
thunder_img = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in thunder]
nums_thunder_punch = len(thunder_punch)
thunder_punch_img = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in thunder_punch]

#Xác định tọa độ điểm trên bàn tay
def position_data(lmlist):
    global wrist, thumb_tip, thumb_mcp, index_mcp, index_tip, midle_mcp, midle_tip, ring_tip, ring_mcp, pinky_tip, pinky_mcp
    wrist = (lmlist[0][0], lmlist[0][1])
    thumb_tip = (lmlist[4][0], lmlist[4][1])
    thumb_mcp = (lmlist[1][0], lmlist[1][1])
    index_mcp = (lmlist[5][0], lmlist[5][1])
    index_tip = (lmlist[8][0], lmlist[8][1])
    midle_mcp = (lmlist[9][0], lmlist[9][1])
    midle_tip = (lmlist[12][0], lmlist[12][1])
    ring_tip = (lmlist[16][0], lmlist[16][1])
    ring_mcp = (lmlist[13][0], lmlist[13][1])
    pinky_tip = (lmlist[20][0], lmlist[20][1])
    pinky_mcp = (lmlist[17][0], lmlist[17][1])

#Tính khoảng cách giữa 2 điểm
def calculate_distance(p1,p2):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    lenght = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2)
    return lenght

#Ghép ảnh vào Background
def transparent(targetImg, x, y, size=None):
    if size is not None:
        targetImg = cv2.resize(targetImg, size)

    newFrame = img.copy()
    b, g, r = cv2.split(targetImg)
    overlay_color = cv2.merge((b, g, r))
    h, w, _ = overlay_color.shape
    roi = newFrame[y:y + h, x:x + w]
    img2gray = cv2.cvtColor(overlay_color, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    newFrame[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return newFrame

while True:
    #Detect hand
    ret, img = video.read()
    img = cv2.flip(img, 1)
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgbimg)
    hand_c = 0
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            hand_c += 1
            #Lấy tọa độ các điểm trên bàn tay
            lmList = []
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                coorx, coory = int(lm.x * w), int(lm.y * h)
                lmList.append([coorx, coory])
            position_data(lmList)
            #Tính khoảng cách các điểm
            dis_thumb = calculate_distance(thumb_tip, wrist)
            dis_index = calculate_distance(index_tip, wrist)
            dis_midle = calculate_distance(midle_tip, wrist)
            dis_ring = calculate_distance(ring_tip, wrist)
            dis_pinky = calculate_distance(pinky_tip, wrist)
            dis_thumb_mcp = calculate_distance(thumb_mcp, wrist)
            dis_index_mcp = calculate_distance(index_mcp, wrist)
            dis_midle_mcp = calculate_distance(midle_mcp, wrist)
            dis_ring_mcp = calculate_distance(ring_mcp, wrist)
            dis_pinky_mcp = calculate_distance(pinky_mcp, wrist)
            dis_thumb_index = calculate_distance(thumb_tip, index_tip)
            dis_thumb_ring = calculate_distance(thumb_tip, ring_tip)
            dis_tip_5f = (dis_thumb + dis_index + dis_midle + dis_ring + dis_pinky) / 5
            dis_mcp_5f = (dis_thumb_mcp + dis_index_mcp + dis_midle_mcp + dis_ring_mcp + dis_pinky_mcp) / 5
            #Tính tỉ lệ khoảng cách giữa các điểm
            ratio_punch =  dis_tip_5f / dis_mcp_5f
            ratio_attack = dis_tip_5f / dis_mcp_5f
            ratio_ball = dis_thumb_index / dis_thumb_ring
            size = 3.0
            #Hiệu ứng nắm đấm sấm sét
            diameter = round(1280*size)
            if (ratio_punch < 1):
                    centerx = 720/2
                    centery = 1280/2
                    x1 = round(centerx - diameter/2)
                    y1 = round(centery - diameter/2)
                    h, w, c = img.shape
                    if x1 < 0:
                        x1 = 0
                    elif x1 > w:
                        x1 = w
                    if y1 < 0:
                        y1 = 0
                    elif y1 > h:
                        y1 = h
                    if x1 + diameter > w:
                        diameter = w - x1
                    if y1 + diameter > h:
                        diameter = h - y1
                    size = diameter, diameter
                    if (diameter != 0):
                        img = transparent(thunder_punch_img[i_punch], x1, y1, size)
                    i_punch = (i_punch + 1) % nums_thunder_punch
            #Tạo hiệu ứng nạp năng lượng
            elif ratio_ball < 1.1 and ratio_ball > 0.9:
                    if energy < 120:
                        energy += 3
                    if energy == 120:
                        attack = True
                        cv2.putText(img, 'Full Energy', (250, 50),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.9,
                                    (0, 255, 0), 2)
                        energy += 0
                    centerx = (midle_mcp[0] + wrist[0])/2
                    centery = (midle_mcp[1] + wrist[1])/2
                    diameter = round(calculate_distance(wrist,index_mcp)*size)
                    x1 = round(centerx - diameter/2)
                    y1 = round(centery - diameter/2)
                    h, w, c = img.shape
                    if x1 < 0:
                        x1 = 0
                    elif x1 > w:
                        x1 = w
                    if y1 < 0:
                        y1 = 0
                    elif y1 > h:
                        y1 = h
                    if x1 + diameter > w:
                        diameter = w - x1
                    if y1 + diameter > h:
                        diameter = h - y1
                    size = diameter, diameter
                    if (diameter != 0):
                        img = transparent(thunderball_img[i_ball], x1, y1, size)
                    i_ball = (i_ball + 1) % nums_thunderball
            # if energy == 120:
            #     attack = True
            #     cv2.putText(img, 'Full Energy', (250, 50),
            #                 cv2.FONT_HERSHEY_COMPLEX, 0.9,
            #                 (0, 255, 0), 2)
            # elif energy >= 123:
            #     energy -= 3
            #Tạo hiệu ứng tung skill
            if hand_c == 2:
                if attack == True:
                    if ratio_attack > 1.8:
                        if energy > 0:
                            energy -= 2
                            print(energy)
                            centerx = 720 / 2
                            centery = 1280 / 2
                            x1 = round(centerx - diameter / 2)
                            y1 = round(centery - diameter / 2)
                            h, w, c = img.shape
                            if x1 < 0:
                                x1 = 0
                            elif x1 > w:
                                x1 = w
                            if y1 < 0:
                                y1 = 0
                            elif y1 > h:
                                y1 = h
                            if x1 + diameter > w:
                                diameter = w - x1
                            if y1 + diameter > h:
                                diameter = h - y1
                            size = diameter, diameter
                            if (diameter != 0):
                                img = transparent(thunder_img[i_thunder], x1, y1, size)
                            i_thunder = (i_thunder + 1) % nums_thunder
                        elif energy == 0:
                            attack = False

    # print(result)
    cv2.imshow("God of thunder",img)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()