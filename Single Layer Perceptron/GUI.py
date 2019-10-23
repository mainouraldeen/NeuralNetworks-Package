from tkinter import *
from tkinter import messagebox
import Model

top = Tk()
top.geometry("1000x500")


########################################################################
# Functions
def collectData():
    feature1 = feature1var.get()
    feature2 = feature2var.get()
    class1 = class1var.get()
    class2 = class2var.get()
    learningRate = LearningRatetextBox.get()
    epochs = EpochstextBox.get()
    bias = biasvar.get()
    '''
    if (class1 == "Iris-setosa"):
        class1 = "001"
    if (class1 == "Iris-versicolor"):
        class1 = "010"
    if (class1 == "Iris-virginica"):
        class1 = "100"

    if (class2 == "Iris-setosa"):
        class2 = "001"
    if (class2 == "Iris-versicolor"):
        class2 = "010"
    if (class2 == "Iris-virginica"):
        class2 = "100" 
    '''
    print(feature1, feature2, class1, class2, learningRate, epochs, bias)


########################################################################
# labels
var = StringVar()
l1 = Label(top, textvariable=var)
var.set("Feature 1")
l1.place(x=50, y=50)

var = StringVar()
l2 = Label(top, textvariable=var)
var.set("Feature 2")
l2.place(x=400, y=50)

var = StringVar()
l3 = Label(top, textvariable=var)
var.set("Class 1")
l3.place(x=50, y=200)

var = StringVar()
l4 = Label(top, textvariable=var)
var.set("Class 2")
l4.place(x=400, y=200)

var = StringVar()
l5 = Label(top, textvariable=var)
var.set("Learning Rate")
l5.place(x=50, y=350)

var = StringVar()
l6 = Label(top, textvariable=var)
var.set("Epochs")
l6.place(x=400, y=350)

var = StringVar()
l7 = Label(top, textvariable=var)
var.set("Bias")
l7.place(x=700, y=350)

########################################################################
# Buttons
drawButton = Button(top, text="Draw Iris Data", command=Model.DrawData)
drawButton.place(x=200, y=450)

runButton = Button(top, text="Run", command=collectData)
runButton.place(x=600, y=450)

########################################################################
# TextBoxs
LearningRatetextBox = Entry(top)
LearningRatetextBox.place(x=140, y=350)
LearningRatetextBox.focus_set()

EpochstextBox = Entry(top)
EpochstextBox.place(x=480, y=350)

########################################################################
# DropBox
feature1DropBox = Frame(top)
feature1DropBox.place(y=50, x=120)
feature1var = StringVar(top)
choices = {'X1', 'X2', 'X3', 'X4'}
feature1var.set('X1')  # set the default option
popupMenu = OptionMenu(feature1DropBox, feature1var, *choices)
popupMenu.grid(row=2, column=1)

feature2DropBox = Frame(top)
feature2DropBox.place(y=50, x=500)
feature2var = StringVar(top)
choices = {'X1', 'X2', 'X3', 'X4'}
feature2var.set('X1')  # set the default option
popupMenu = OptionMenu(feature2DropBox, feature2var, *choices)
popupMenu.grid(row=2, column=1)

class1DropBox = Frame(top)
class1DropBox.place(y=200, x=120)
class1var = StringVar(top)
choices = {'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'}
class1var.set('Iris-setosa')  # set the default option
popupMenu = OptionMenu(class1DropBox, class1var, *choices)
popupMenu.grid(row=2, column=1)

class2DropBox = Frame(top)
class2DropBox.place(y=200, x=500)
class2var = StringVar(top)
choices = {'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'}
class2var.set('Iris-setosa')  # set the default option
popupMenu = OptionMenu(class2DropBox, class2var, *choices)
popupMenu.grid(row=2, column=1)

biasDropBox = Frame(top)
biasDropBox.place(y=350, x=750)
biasvar = StringVar(top)
choices = {'0', '1'}
biasvar.set('1')  # set the default option
popupMenu = OptionMenu(biasDropBox, biasvar, *choices)
popupMenu.grid(row=2, column=1)
########################################################################


top.mainloop()
