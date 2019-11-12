from tkinter import *

import Model

top = Tk()
top.geometry("700x300")


# Functions
def collectData():
    numOfHiddenLayers = numOfHiddenLayerstextBox.get()
    numOfNeurons = numOfNeuornstextBox.get().split(",")
    learningRate = LearningRatetextBox.get()
    epochs = EpochstextBox.get()
    bias = biasvar.get()
    activationFunction = activationFun.get()

    Model.main(numOfHiddenLayers, numOfNeurons, learningRate, epochs, bias, activationFunction)


# labels
# region
var = StringVar()
l1 = Label(top, textvariable=var)
var.set("Enter number of hidden layers:")
l1.place(x=50, y=50)

var = StringVar()
l2 = Label(top, textvariable=var)
var.set("Enter number of neurons:")
l2.place(x=400, y=50)

var = StringVar()
l5 = Label(top, textvariable=var)
var.set("Learning Rate:")
l5.place(x=50, y=100)

var = StringVar()
l6 = Label(top, textvariable=var)
var.set("Epochs:")
l6.place(x=400, y=100)

var = StringVar()
l7 = Label(top, textvariable=var)
var.set("Bias:")
l7.place(x=400, y=150)

var = StringVar()
l8 = Label(top, textvariable=var)
var.set("Activation Function:")
l8.place(x=50, y=150)
# endregion

# Buttons
# region
# drawButton = Button(top, text="Draw Iris Data", command=Model.drawIrisData)
# drawButton.place(x=200, y=450)

runButton = Button(top, text="Run", command=collectData)
runButton.place(x=350, y=200)
# endregion

# TextBoxs
# region
numOfHiddenLayerstextBox = Entry(top)
numOfHiddenLayerstextBox.place(x=220, y=50)
numOfHiddenLayerstextBox.focus_set()
numOfHiddenLayerstextBox.insert(0, "2")

numOfNeuornstextBox = Entry(top)
numOfNeuornstextBox.place(x=550, y=50)
numOfNeuornstextBox.insert(0, "2,5")

LearningRatetextBox = Entry(top)
LearningRatetextBox.place(x=140, y=100)
LearningRatetextBox.focus_set()
LearningRatetextBox.insert(0, "0.01")

EpochstextBox = Entry(top)
EpochstextBox.place(x=460, y=100)
EpochstextBox.insert(0, "20000")

# endregion

# DropBox
# region

biasDropBox = Frame(top)
biasDropBox.place(x=460, y=150)
biasvar = StringVar(top)
choices = {'0', '1'}
biasvar.set('1')  # set the default option
popupMenu = OptionMenu(biasDropBox, biasvar, *choices)
popupMenu.grid(row=2, column=1)

activationFunDropBox = Frame(top)
activationFunDropBox.place(y=150, x=180)
activationFun = StringVar(top)
choices = {"Sigmoid", "Tanh"}
activationFun.set("Sigmoid")  # set the default option
popupMenu = OptionMenu(activationFunDropBox, activationFun, *choices)
popupMenu.grid(row=2, column=1)
# endregion

# main
top.mainloop()
