import cv2
import numpy as np

# wykonuje operacje skalowania obrazu
def skalowanie_obrazu(skala, image): 
    h = image.shape[0]
    w = image.shape[1]
    width_scaled = int(w*skala)
    height_scaled = int(h*skala)
    resized = cv2.resize(image, (width_scaled, height_scaled))
    return resized

# wykonuje szereg operacji przygotowywujacych obraz do dalszych operacji
def przygotowanie_obrazu(resized):
    # konwersja na czarno_bialy
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # usuniecie szumow
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    # progowanie
    thres = cv2.threshold(blurred, 125, 255, cv2.THRESH_BINARY_INV)[1]
    # dylatacja
    dylatacja = cv2.dilate(thres, kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations = 1)
    # erozja
    erozja = cv2.erode(dylatacja, kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations = 1)
    # dylatacja 2
    dylatacja_2 = cv2.dilate(erozja, kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations = 1)
    # zamkniecie
    kernel = np.ones((4,4),np.uint8)
    closing = cv2.morphologyEx(dylatacja_2, cv2.MORPH_CLOSE, kernel)
    # erozja 2
    erozja_2 = cv2.erode(closing, kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2)), iterations = 1)
    # wykrywanie krawedzi
    obr_kraw = cv2.Canny(erozja_2, 100, 255)
    return obr_kraw

# sortuje kontury, tak zeby numeracja byla po kolei od lewej do prawej
def sortowanie_konturow(cnts):
	reverse = False
	i = 0 
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
	return (cnts, boundingBoxes)

# znajduje obiekty i rysuje prostokat wokol nich
def znalezienie_obszaru(image, image_p):
    contours = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    (contours, boundingBoxes) = sortowanie_konturow(contours)
    rects = [[0] * 4] * len(contours)
    for i in range (0, len(contours)):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        rects[i] = cv2.boundingRect(cnt)
    return image, rects, contours

# znajduje obiekty i rysuje prostokat wokol nich, ale z najmniejszym polem, tzn. moga byc obrocone        
def obszar_obrocony(image, image_p):
    contours = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    rects = [[0] * 4] * len(contours)
    for i in range (0, len(contours)):
        cnt = contours[i]
        rect = cv2.minAreaRect(cnt)
        rects[i] = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
    return image, rects, contours

# szuka maksymalnej wysokosci prostokatow
def max_wysokosc(rects, czy_minarearect):
    h = 0
    if czy_minarearect==1:
        for i in range(0,len(rects)):
            if h<rects[i][1][1]:
                h = rects[i][1][1]
    elif czy_minarearect==0:
        for i in range(0,len(rects)):
            if h<rects[i][3]:
                h = rects[i][3]
    return h

# zwraca odleglosc y od obszaru
def odl_od_gory(rects, czy_minarearect):
    y_min = 1000
    for i in rects:
        if y_min>i[1]:
            y_min = i[1]
    return y_min
       
# na podstawie maksymalnej wysokosci prostokatow metoda dostosowuje rozmiar obrazu
def ten_sam_rozmiar(image, maxheight):
    oczekiwana_wysokosc = 100
    skala = oczekiwana_wysokosc/maxheight
    image = skalowanie_obrazu(skala, image)
    return image

# ucina obszar obrazu, na ktorym nie ma rownania
def bez_tla(image, max_h, min_y, x, x_2):
    uciety = image[min_y:min_y+max_h,x:x_2]
    return uciety

# transformata Hougha dla okręgów - nie uzywana
def tr_okregi(image, image_wyj):
    circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,10,param1=50,param2=20,minRadius=5,maxRadius=30)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # rysowanie obwodu okregu
        cv2.circle(image_wyj,(i[0],i[1]),i[2],(0,255,0),2)
        # rysowanie srodka okregu
        cv2.circle(image_wyj,(i[0],i[1]),2,(0,0,255),3)
    return image_wyj
        
# tranformata Hougha P
def tr_linie(image):
    minLineLength = 10
    maxLineGap = 3
    lines = cv2.HoughLinesP(image,1,np.pi/180,10,minLineLength=minLineLength,maxLineGap=maxLineGap)
    return lines

def spr_cyfr_i_znakow(image_skalowane, rects):
    # tablica do przechawywania wykrytych cyfr i znakow
    tab = [0]*len(rects)
    """for j in range(len(rects)):
        tab[j][0]=1"""
    for i in range(len(rects)):
        (x, y, w, h) = rects[i]
        image_w_1 = image_skalowane[y-5:y+h+5,x-5:x+w+5]
        image_1 = przygotowanie_obrazu(image_w_1)
        # okrelenie czy w obszarze wystepuje kreska u gory
        linie = tr_linie(image_1.copy())
        czy_kreska_u_gory=0
        if linie is not None:
            for line in linie:
                x1, y1, x2, y2 = line[0]
                if (x1<15 and y1<15 and x2>30 and y2<15):
                    czy_kreska_u_gory=1
        # obliczenie pola prostokata
        area_rect = w*h
        # okreslenie dlugosci konturu
        contours_2 = cv2.findContours(image_1.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
        area_cnt = cv2.contourArea(contours_2[0])
        l_cnt = cv2.arcLength(contours_2[0],True)
        # obliczenie wspolczynnika smuklosci
        wsp_s = 0
        if area_cnt>10:
            wsp_s = l_cnt*l_cnt/(4*3.14*area_cnt)
        # okreslenie liczby konturow wewnetrznych
        contours_3 = cv2.findContours(image_1.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)[1] 
        count = len(contours_3)/2-1
        # znalezienie kata pochylenia prostakata o najmniejszym polu obszaru
        kat = obszar_obrocony(image_1.copy(),image_1.copy())[1][0][2]
        # obliczenie stosunku szerokosci do dlugosci prostokata
        st_sz_dl = h/w
        # sprawdzenie czy istnieje kontur znajdujacy sie tylko na gorze lub tylko na dolu obszaru
        czy_na_dole=0
        czy_na_gorze=0
        for cn in contours_3:
            x_1,y_1,w_1,h_1 = cv2.boundingRect(cn)
            if (y_1<50 and w_1*h_1<1800):
                czy_na_gorze = 1
            if (y_1>=50 and w_1*h_1<1800):
                czy_na_dole = 1
        # warunki na cyfry
        if area_rect>2700:
            #if tab[i][0]==1:
                if(area_cnt>2000 and count==1):
                    print(str(0))
                    tab[i]="0"
                elif count==2:
                    print(str(8))
                    tab[i]="8"
                elif (count==1 and czy_na_gorze==1):
                    print(str(9))
                    tab[i]="9"
                elif (count==1 and czy_na_dole==1):
                    print(str(6))
                    tab[i]="6"
                elif czy_kreska_u_gory==1:
                    print(str(5))
                    tab[i]="5"
                elif (wsp_s<5.65 and wsp_s>1.8):
                    print(str(1))
                    tab[i]="1"
                elif (l_cnt>220 and l_cnt<300):
                    print(str(4))
                    tab[i]="4"
                elif (l_cnt>310 and l_cnt<360):
                    print(str(7))
                    tab[i]="7"
                elif (wsp_s<9 and wsp_s>6):
                    print(str(2))
                    tab[i]="2"
                elif (wsp_s<15 and wsp_s>9):
                    print(str(3))
                    tab[i]="3"
                else: 
                    print("?")
        # warunki na znaki
        if area_rect<=2700:
            #if tab[i][0]==1:
                if st_sz_dl<0.3:
                    image_w_2 = image_skalowane[y-40:y+40,x-5:x+w+5]
                    image_2 = przygotowanie_obrazu(image_w_2)
                    contours_4 = cv2.findContours(image_2.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1] 
                    count_1 = len(contours_4)/2
                    if count_1==1:
                        print("-")
                        tab[i]="-"
                    elif count_1==2:
                        print("=")
                        tab[i]="="
                elif area_rect<450:
                    image_w_2 = image_skalowane[y-60:y+60,x-5:x+w+5]
                    image_2 = przygotowanie_obrazu(image_w_2)
                    contours_4 = cv2.findContours(image_2.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1] 
                    count_1 = len(contours_4)/2
                    if count_1==1:
                        print("*")
                        tab[i]="*"
                    elif count_1==2:
                        print(":")
                        tab[i]=":"
                elif((kat>38 and kat<52)or(kat>-52 and kat<-38)):
                    print("+")
                    tab[i]="+"
                else:
                    print("?")
    return tab

# deklaracje obrazow
testowe_cyfry = cv2.imread("obrazy/testowe_cyfry/testowe_cyfry.jpg")
dzialanie_dod2= cv2.imread("obrazy/dzialania/dodawanie (2).jpg")      
dzialanie_odj1 = cv2.imread("obrazy/dzialania/odejmowanie (1).jpg")
dzialanie_mnz2 = cv2.imread("obrazy/dzialania/mnozenie (2).jpg")
dzialanie_dzl1 = cv2.imread("obrazy/dzialania/dzielenie (1).jpg")

# skalowanie obrazow
testowe_cyfry_skalowane = skalowanie_obrazu(0.1, testowe_cyfry)
dzialanie_dod2_skalowane = skalowanie_obrazu(0.1, dzialanie_dod2)
dzialanie_odj1_skalowane = skalowanie_obrazu(0.1, dzialanie_odj1)
dzialanie_mnz2_skalowane = skalowanie_obrazu(0.1, dzialanie_mnz2)
dzialanie_dzl1_skalowane = skalowanie_obrazu(0.1, dzialanie_dzl1)

# przygotowanie obrazow
testowe_cyfry_1 = przygotowanie_obrazu(testowe_cyfry_skalowane)
dzialanie_dod2_1 = przygotowanie_obrazu(dzialanie_dod2_skalowane) 
dzialanie_odj1_1 = przygotowanie_obrazu(dzialanie_odj1_skalowane)  
dzialanie_mnz2_1 = przygotowanie_obrazu(dzialanie_mnz2_skalowane)
dzialanie_dzl1_1 = przygotowanie_obrazu(dzialanie_dzl1_skalowane) 

# stworzenie kopii
testowe_kopia = testowe_cyfry_1.copy()
dzialanie_dod2_kopia = dzialanie_dod2_1.copy()
dzialanie_odj1_kopia = dzialanie_odj1_1.copy()
dzialanie_mnz2_kopia = dzialanie_mnz2_1.copy()
dzialanie_dzl1_kopia = dzialanie_dzl1_1.copy()

# znalezioenie konturow
testowe_kontury = znalezienie_obszaru(testowe_cyfry_1.copy(), testowe_cyfry_skalowane)[2]
dzialanie_dod2_kontury = znalezienie_obszaru(dzialanie_dod2_1, dzialanie_dod2_skalowane)[2]
dzialanie_odj1_kontury = znalezienie_obszaru(dzialanie_odj1_1, dzialanie_odj1_skalowane)[2]
dzialanie_mnz2_kontury = znalezienie_obszaru(dzialanie_mnz2_1, dzialanie_mnz2_skalowane)[2]
dzialanie_dzl1_kontury = znalezienie_obszaru(dzialanie_dzl1_1, dzialanie_dzl1_skalowane)[2]

# znalezioenie prostakatow
rects_testowe = znalezienie_obszaru(testowe_cyfry_1.copy(), testowe_cyfry_skalowane)[1]
rects_dod2 = znalezienie_obszaru(dzialanie_dod2_1.copy(), dzialanie_dod2_skalowane)[1]
rects_odj1 = znalezienie_obszaru(dzialanie_odj1_1.copy(), dzialanie_odj1_skalowane)[1]
rects_mnz2 = znalezienie_obszaru(dzialanie_mnz2_1.copy(), dzialanie_mnz2_skalowane)[1]
rects_dzl1 = znalezienie_obszaru(dzialanie_dzl1_1.copy(), dzialanie_dzl1_skalowane)[1]

# znalezienie najwyzszego prostokata na obrazie
max_height_testowe = max_wysokosc(rects_testowe,czy_minarearect=0)
max_height_dod2 = max_wysokosc(rects_dod2,czy_minarearect=0)
max_height_odj1 = max_wysokosc(rects_odj1,czy_minarearect=0)
max_height_mnz2 = max_wysokosc(rects_mnz2,czy_minarearect=0)
max_height_dzl1 = max_wysokosc(rects_dzl1,czy_minarearect=0)

# okreslenie minimalnej odleglosci cyfr od gory obrazu
min_y_testowe = odl_od_gory(rects_testowe, 0)
min_y_dod2 = odl_od_gory(rects_dod2, 0)
min_y_odj1 = odl_od_gory(rects_odj1, 0)
min_y_mnz2 = odl_od_gory(rects_mnz2, 0)
min_y_dzl1 = odl_od_gory(rects_dzl1, 0)

# usuniecie zbednych obszarow
testowe_1_bez_tla = bez_tla(testowe_cyfry_1, max_height_testowe, min_y_testowe, rects_testowe[0][0], rects_testowe[len(rects_testowe)-1][0]+rects_testowe[len(rects_testowe)-1][2])
testowe_skalowane_bez_tla = bez_tla(testowe_cyfry_skalowane, max_height_testowe, min_y_testowe, rects_testowe[0][0], rects_testowe[len(rects_testowe)-1][0]+rects_testowe[len(rects_testowe)-1][2]) 
testowe_kopia_bez_tla = bez_tla(testowe_kopia, max_height_testowe, min_y_testowe, rects_testowe[0][0], rects_testowe[len(rects_testowe)-1][0]+rects_testowe[len(rects_testowe)-1][2])

dod2_1_bez_tla = bez_tla(dzialanie_dod2_1, max_height_dod2, min_y_dod2, rects_dod2[0][0], rects_dod2[len(rects_dod2)-1][0]+rects_dod2[len(rects_dod2)-1][2])
dod2_skalowane_bez_tla = bez_tla(dzialanie_dod2_skalowane, max_height_dod2, min_y_dod2, rects_dod2[0][0], rects_dod2[len(rects_dod2)-1][0]+rects_dod2[len(rects_dod2)-1][2]) 
dod2_kopia_bez_tla = bez_tla(dzialanie_dod2_kopia, max_height_dod2, min_y_dod2, rects_dod2[0][0], rects_dod2[len(rects_dod2)-1][0]+rects_dod2[len(rects_dod2)-1][2])

odj1_1_bez_tla = bez_tla(dzialanie_odj1_1, max_height_odj1, min_y_odj1, rects_odj1[0][0], rects_odj1[len(rects_odj1)-1][0]+rects_odj1[len(rects_odj1)-1][2])
odj1_skalowane_bez_tla = bez_tla(dzialanie_odj1_skalowane, max_height_odj1, min_y_odj1, rects_odj1[0][0], rects_odj1[len(rects_odj1)-1][0]+rects_odj1[len(rects_odj1)-1][2]) 
odj1_kopia_bez_tla = bez_tla(dzialanie_odj1_kopia, max_height_odj1, min_y_odj1, rects_odj1[0][0], rects_odj1[len(rects_odj1)-1][0]+rects_odj1[len(rects_odj1)-1][2])

mnz2_1_bez_tla = bez_tla(dzialanie_mnz2_1, max_height_mnz2, min_y_mnz2, rects_mnz2[0][0], rects_mnz2[len(rects_mnz2)-1][0]+rects_mnz2[len(rects_mnz2)-1][2])
mnz2_skalowane_bez_tla = bez_tla(dzialanie_mnz2_skalowane, max_height_mnz2, min_y_mnz2, rects_mnz2[0][0], rects_mnz2[len(rects_mnz2)-1][0]+rects_mnz2[len(rects_mnz2)-1][2]) 
mnz2_kopia_bez_tla = bez_tla(dzialanie_mnz2_kopia, max_height_mnz2, min_y_mnz2, rects_mnz2[0][0], rects_mnz2[len(rects_mnz2)-1][0]+rects_mnz2[len(rects_mnz2)-1][2])

dzl1_1_bez_tla = bez_tla(dzialanie_dzl1_1, max_height_dzl1, min_y_dzl1, rects_dzl1[0][0], rects_dzl1[len(rects_dzl1)-1][0]+rects_dzl1[len(rects_dzl1)-1][2])
dzl1_skalowane_bez_tla = bez_tla(dzialanie_dzl1_skalowane, max_height_dzl1, min_y_dzl1, rects_dzl1[0][0], rects_dzl1[len(rects_dzl1)-1][0]+rects_dzl1[len(rects_dzl1)-1][2]) 
dzl1_kopia_bez_tla = bez_tla(dzialanie_dzl1_kopia, max_height_dzl1, min_y_dzl1, rects_dzl1[0][0], rects_dzl1[len(rects_dzl1)-1][0]+rects_dzl1[len(rects_dzl1)-1][2])

# dopasowanie rozmiaru
testowe_cyfry_1 = ten_sam_rozmiar(testowe_cyfry_1, max_height_testowe)
testowe_kopia = ten_sam_rozmiar(testowe_kopia, max_height_testowe)
testowe_cyfry_skalowane = ten_sam_rozmiar(testowe_cyfry_skalowane, max_height_testowe)

dzialanie_dod2_1 = ten_sam_rozmiar(dzialanie_dod2_1, max_height_dod2)
dzialanie_dod2_kopia = ten_sam_rozmiar(dzialanie_dod2_kopia, max_height_dod2)
dzialanie_dod2_skalowane = ten_sam_rozmiar(dzialanie_dod2_skalowane, max_height_dod2)

dzialanie_odj1_1 = ten_sam_rozmiar(dzialanie_odj1_1, max_height_odj1)
dzialanie_odj1_kopia = ten_sam_rozmiar(dzialanie_odj1_kopia, max_height_odj1)
dzialanie_odj1_skalowane = ten_sam_rozmiar(dzialanie_odj1_skalowane, max_height_odj1)

dzialanie_mnz2_1 = ten_sam_rozmiar(dzialanie_mnz2_1, max_height_mnz2)
dzialanie_mnz2_kopia = ten_sam_rozmiar(dzialanie_mnz2_kopia, max_height_mnz2)
dzialanie_mnz2_skalowane = ten_sam_rozmiar(dzialanie_mnz2_skalowane, max_height_mnz2)

dzialanie_dzl1_1 = ten_sam_rozmiar(dzialanie_dzl1_1, max_height_dzl1)
dzialanie_dzl1_kopia = ten_sam_rozmiar(dzialanie_dzl1_kopia, max_height_dzl1)
dzialanie_dzl1_skalowane = ten_sam_rozmiar(dzialanie_dzl1_skalowane, max_height_dzl1)

# stworzenie konturow dla zeskalowanego obrazu - wys. cyfry: 100
testowe_kontury_2 = znalezienie_obszaru(testowe_kopia.copy(), testowe_cyfry_skalowane)[2]
dod2_kontury_2 = znalezienie_obszaru(dzialanie_dod2_kopia.copy(), dzialanie_dod2_skalowane)[2]
odj1_kontury_2 = znalezienie_obszaru(dzialanie_odj1_kopia.copy(), dzialanie_odj1_skalowane)[2]
mnz2_kontury_2 = znalezienie_obszaru(dzialanie_mnz2_kopia.copy(), dzialanie_mnz2_skalowane)[2]
dzl1_kontury_2 = znalezienie_obszaru(dzialanie_dzl1_kopia.copy(), dzialanie_dzl1_skalowane)[2]

# znalezioenie prostakatow dla zeskalowanego obrazu
rects_testowe_2 = znalezienie_obszaru(testowe_kopia.copy(), testowe_cyfry_skalowane)[1]
rects_dod2_2 = znalezienie_obszaru(dzialanie_dod2_kopia.copy(), dzialanie_dod2_skalowane)[1]
rects_odj1_2 = znalezienie_obszaru(dzialanie_odj1_kopia.copy(), dzialanie_odj1_skalowane)[1]
rects_mnz2_2 = znalezienie_obszaru(dzialanie_mnz2_kopia.copy(), dzialanie_mnz2_skalowane)[1]
rects_dzl1_2 = znalezienie_obszaru(dzialanie_dzl1_kopia.copy(), dzialanie_dzl1_skalowane)[1]

"""# funkcja rozpoznajaca cyfry i znaki na obrazie
spr_cyfr_i_znakow(testowe_cyfry_skalowane, rects_testowe_2)

# wyswietlenie obrazu
cv2.imshow("zera", testowe_skalowane_bez_tla)
cv2.waitKey(0)"""

tab = spr_cyfr_i_znakow(dzialanie_dod2_skalowane, rects_dod2_2)

cv2.imshow("zera", dzialanie_dod2_skalowane)
cv2.waitKey(0)

"""spr_cyfr_i_znakow(dzialanie_odj1_skalowane, rects_odj1_2)

cv2.imshow("zera", dzialanie_odj1_skalowane)
cv2.waitKey(0)

spr_cyfr_i_znakow(dzialanie_mnz2_skalowane, rects_mnz2_2)

cv2.imshow("zera", dzialanie_mnz2_skalowane)
cv2.waitKey(0)

spr_cyfr_i_znakow(dzialanie_dzl1_skalowane, rects_dzl1_2)

cv2.imshow("zera", dzialanie_dzl1_skalowane)
cv2.waitKey(0)
"""

equation = ""

for i in range (0, len(tab)):
    if(tab[i]==tab[i-1] and tab[i]=="*"):
        continue
    if(tab[i]=="="):
        break
    equation = equation + tab[i]
    
result = eval(equation)
print("Equation: "+ str(equation)+"=")
print("Result: "+ str(result))