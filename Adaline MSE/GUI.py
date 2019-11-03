from tkinter import *

import Model

top = Tk()
top.geometry("900x500")


# Functions
def collectData():
    feature1 = feature1var.get()
    feature2 = feature2var.get()
    class1 = class1var.get()
    class2 = class2var.get()
    learningRate = LearningRatetextBox.get()
    epochs = EpochstextBox.get()
    bias = biasvar.get()
    testFeature1 = X1TesttextBox.get()
    testFeature2 = X2TesttextBox.get()
    MSEthreshold = MSEtextBox.get()

    if class1 > class2:
        class1, class2 = class2, class1
    Model.main(feature1, feature2, class1, class2, testFeature1, testFeature2, learningRate, epochs, bias, MSEthreshold)


# labels
# region
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

var = StringVar()
l8 = Label(top, textvariable=var)
var.set("Test feature 1")
l8.place(x=50, y=400)

var = StringVar()
l9 = Label(top, textvariable=var)
var.set("Test feature 2")
l9.place(x=390, y=400)

var = StringVar()
l10 = Label(top, textvariable=var)
var.set("MSE")
l10.place(x=50, y=150)
# endregion

# Buttons
# region
drawButton = Button(top, text="Draw Iris Data", command=Model.drawIrisData)
drawButton.place(x=200, y=450)

runButton = Button(top, text="Run", command=collectData)
runButton.place(x=600, y=450)
# endregion

# TextBoxs
# region
LearningRatetextBox = Entry(top)
LearningRatetextBox.place(x=140, y=350)
LearningRatetextBox.focus_set()
LearningRatetextBox.insert(0, "0.001")

EpochstextBox = Entry(top)
EpochstextBox.place(x=480, y=350)
EpochstextBox.insert(0, "10000")

X1TesttextBox = Entry(top)
X1TesttextBox.place(x=140, y=400)
X1TesttextBox.insert(0, "5")

X2TesttextBox = Entry(top)
X2TesttextBox.place(x=480, y=400)
X2TesttextBox.insert(0, "3")

MSEtextBox = Entry(top)
MSEtextBox.place(x=100, y=150)
MSEtextBox.insert(0, "0.005")
# endregion

# DropBox
# region
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
feature2var.set('X3')  # set the default option
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
class2var.set('Iris-virginica')  # set the default option
popupMenu = OptionMenu(class2DropBox, class2var, *choices)
popupMenu.grid(row=2, column=1)

biasDropBox = Frame(top)
biasDropBox.place(y=350, x=750)
biasvar = StringVar(top)
choices = {'0', '1'}
biasvar.set('1')  # set the default option
popupMenu = OptionMenu(biasDropBox, biasvar, *choices)
popupMenu.grid(row=2, column=1)
# endregion

# main
top.mainloop()
