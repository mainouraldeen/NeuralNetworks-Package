from tkinter import *
from tkinter import messagebox
import Model


top =Tk()
top.geometry("1000x500")
########################################################################
#labels
var= StringVar()
l1=Label(top, textvariable=var)
var.set("Feature 1")
l1.place(x=50,y=50)

var= StringVar()
l2=Label(top, textvariable=var)
var.set("Feature 2")
l2.place(x=400,y=50)

var= StringVar()
l3=Label(top, textvariable=var)
var.set("Class 1")
l3.place(x=50,y=200)

var= StringVar()
l4=Label(top, textvariable=var)
var.set("Class 2")
l4.place(x=400,y=200)

var= StringVar()
l5=Label(top, textvariable=var)
var.set("Learning Rate")
l5.place(x=50,y=350)

var= StringVar()
l6=Label(top, textvariable=var)
var.set("Epochs")
l6.place(x=400,y=350)

var= StringVar()
l7=Label(top, textvariable=var)
var.set("Bias")
l7.place(x=700,y=350)

########################################################################
#Buttons
drawButton=Button(top, text="Draw Iris Data",command = Model.DrawIrisData)
drawButton.place(x=200,y=450)

runButton=Button(top, text="Run",command = Model.perceptron)
runButton.place(x=600,y=450)

########################################################################
#TextBoxs
LearningRatetextBox = Entry(top)
LearningRatetextBox.place(x=140,y=350)

EpochstextBox = Entry(top)
EpochstextBox.place(x=480,y=350)

########################################################################
#DropBox
feature1DropBox = Frame(top)
feature1DropBox.place(y = 50, x = 120)
var = StringVar(top)
choices = { 'X1','X2','X3','X4'}
var.set('X1') # set the default option
popupMenu = OptionMenu(feature1DropBox, var, *choices)
popupMenu.grid(row = 2, column =1)

feature2DropBox = Frame(top)
feature2DropBox.place(y = 50, x = 500)
var = StringVar(top)
choices = { 'X1','X2','X3','X4'}
var.set('X1') # set the default option
popupMenu = OptionMenu(feature2DropBox, var, *choices)
popupMenu.grid(row = 2, column =1)

class1DropBox = Frame(top)
class1DropBox.place(y = 200, x = 120)
var = StringVar(top)
choices = { 'Iris-setosa','Iris-versicolor','Iris-virginica'}
var.set('Iris-setosa') # set the default option
popupMenu = OptionMenu(class1DropBox, var, *choices)
popupMenu.grid(row = 2, column =1)

class2DropBox = Frame(top)
class2DropBox.place(y = 200, x = 500)
var = StringVar(top)
choices = { 'Iris-setosa','Iris-versicolor','Iris-virginica'}
var.set('Iris-setosa') # set the default option
popupMenu = OptionMenu(class2DropBox, var, *choices)
popupMenu.grid(row = 2, column =1)

biasDropBox = Frame(top)
biasDropBox.place(y = 350, x = 750)
var = StringVar(top)
choices = { '0','1'}
var.set('1') # set the default option
popupMenu = OptionMenu(biasDropBox, var, *choices)
popupMenu.grid(row = 2, column =1)
########################################################################


top.mainloop()
