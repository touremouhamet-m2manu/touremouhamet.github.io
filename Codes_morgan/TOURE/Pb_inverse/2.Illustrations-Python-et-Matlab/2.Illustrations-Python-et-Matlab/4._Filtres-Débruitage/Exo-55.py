from turtle import forward, backward, left, right, penup, pendown, done

a=10
angle = 37

i=0
while (i<50):
    i+=1
    forward(a)
    left(angle)
    a=a*1.05

done()