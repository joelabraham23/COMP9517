import cv2

def haarCascade(file):

    cascade_penguin = cv2.CascadeClassifier('cascade/cascadePenguin.xml')
    cascade_turtle = cv2.CascadeClassifier('cascade/cascadeTurtle.xml')

    img = cv2.imread(file)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    penguin_rect = cascade_penguin.detectMultiScale(gray_img, scaleFactor=6, minNeighbors=24, minSize=(75,75))
    turtle_rect = cascade_turtle.detectMultiScale(gray_img, scaleFactor=3, minNeighbors=6, minSize=(75,75))

    peng = len(penguin_rect)
    turt = len(turtle_rect)

    if peng > turt:
        return "penguin", peng/(peng + turt)

    if turt > peng:
        return "turtle", turt/(peng + turt)

    return "penguin", 0

def main():
    pass

if __name__ == "__main__":
    main()
