import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

from sklearn import tree


from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from tkinter import ttk

croop=[]
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
root = Tk()
root.title('Crop and Yield Prediction using ML')
root.geometry('850x650')
root.configure(background="gray")

var = StringVar()
label = Label( root, textvariable = var,font=('arial',20,'bold'),bd=20,background="gray")
var.set("Crop and Yield Prediction using Machine Learning")
label.grid(row=0,columnspan=6) 

def train_file():
     root1=Tk()
     root1.title("login page")
     root1.geometry('600x500')
     root1.configure(background="gray")
     def login():
         user = E.get()
         password = E1.get()
         admin_login(user,password)
     L=Label(root1, text = "Username",bd=8,background="gray",height=1,padx=16,pady=16,font=('arial',16,'bold'),width=10,).grid(row = 0,column=0)
     E=Entry(root1)
     E.grid(row = 0, column = 1)
     L1=Label(root1, text = "Password",bd=8,background="gray",height=1,padx=16,pady=16,font=('arial',16,'bold'),width=10,).grid(row = 1,column=0)
     E1=Entry(root1,show="*")
     E1.grid(row = 1, column = 1)
     B1=Button(root1,text="Login",width=4,height=1,command=login,bd=8,background="gray")
     B1.grid(row = 2, column = 1)
     root1.mainloop()

def admin_login(user,password):
     #print(user,password)
     if user == "admin" and password == "admin":
         root3 = Tk()
         root3.title('choose file')
         root3.geometry('600x300')
         root3.configure(background="gray")
         E2=Button(root3,text="Browse file",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="gray",command=OpenFile_train)
         E2.place(x=200,y=100)
         
         
         root3.mainloop()  
     else:
         root3 = Tk()
         root3.title('ERROR')
         L2 = Label(root3, text = "user name and password is wrong",font=('arial',16,'bold'),fg='red').grid(row = 2)
         root3.mainloop()

def OpenFile_train():
    name = askopenfilename(initialdir="C:/Users/Documents/Programming/tkinter/",filetypes =(("csv file", "*.csv"),("All Files","*.*")),
                           title = "Choose a file.")
    try:
        with open(name,'r') as UseFile:
          train(name)
    except FileNotFoundError:
         print("No file exists")

sv = ''
def train(filename): 
    global sv,file1
    file1=filename
    data = pd.read_csv(filename)
    data.info()
    data = data.drop([' Production'],axis=1)
    
    for i in data['Crop']:
            if i not in croop:
                croop.append(i)


    le = preprocessing.LabelEncoder()
    
    Crop = le.fit_transform(data.Crop) 

    testing = data 

    testing['Crop'] = Crop



    predictors = data.drop("Crop",axis=1)
    target = data["Crop"]
    
    X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


    
    sv = tree.DecisionTreeClassifier()
    sv.fit(X_train, Y_train)

    Y_pred_svm = sv.predict(X_test)




    score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
    
    print("The accuracy score achieved using Decision tree is: "+str(score_svm)+" %")
    print(confusion_matrix(Y_test, Y_pred_svm))
    label_1 = ttk.Label(root, text ='Accuracy = '+str(score_svm),font=("Helvetica", 16),background="gray")
    label_1.grid(row=2,column=0)
    
def predict():
    
    root10 = Tk()
    root10.title('Predict')
    root10.geometry('850x650')
    root10.configure(background="gray")
    
    label_1 = ttk.Label(root10, text ='RainFall',font=("Helvetica", 16),background="gray")
    label_1.grid(row=0,column=0)
    
    Entry_1 = Entry(root10)
    Entry_1.grid(row=0,column=1)
    
    label_2 = ttk.Label(root10, text = 'Temp',font=("Helvetica", 16),background="gray")
    label_2.grid(row=1,column=0)
    
    Entry_2 = Entry(root10)
    Entry_2.grid(row=1,column=1)
    
    label_3 = ttk.Label(root10, text = 'PH',font=("Helvetica", 16,),background="gray")
    label_3.grid(row=2,column=0)
    
    Entry_3 = Entry(root10)
    Entry_3.grid(row=2,column=1)
    
    def acc_pH():
        pH=Entry_3.get()
        rain=Entry_1.get()
        temp=Entry_2.get()  
        flag=0
        print(pH,rain,temp)
        if float(pH)<0 or float(pH)>=14:
            flag+=100
        if float(rain)<0 or float(rain)>=1500:
            flag+=10
        if float(temp)<0 or float(temp)>=60:
            flag+=1
        print(flag)
        if(flag==1):
            label_2 = Label(root10, text = "wrong temp value",font=('arial',16,'bold'),fg='red').grid(row = 1)
            predict()  
        if(flag==10):
            label_1 = Label(root10, text = "wrong rain value",font=('arial',16,'bold'),fg='red').grid(row = 0,column=0)
            predict()
        if(flag==100):
            label_3 = Label(root10, text = "wrong pH value",font=('arial',16,'bold'),fg='red').grid(row = 2)
            predict()
        if(flag==101):
            label_3 = Label(root10, text = "wrong pH value",font=('arial',16,'bold'),fg='red').grid(row = 2)
            label_2 = Label(root10, text = "wrong temp value",font=('arial',16,'bold'),fg='red').grid(row = 1)
            predict()
        if(flag==110):
            label_3 = Label(root10, text = "wrong pH value",font=('arial',16,'bold'),fg='red').grid(row = 2)
            label_1 = Label(root10, text = "wrong rain value",font=('arial',16,'bold'),fg='red').grid(row = 0,column = 0)
            predict()
        if(flag==11):
            label_2 = Label(root10, text = "wrong temp value",font=('arial',16,'bold'),fg='red').grid(row = 1)
            label_1 = Label(root10, text = "wrong rain value",font=('arial',16,'bold'),fg='red').grid(row = 0,column = 0)
            predict()
        if(flag==111):
            label_2 = Label(root10, text = "wrong temp value",font=('arial',16,'bold'),fg='red').grid(row = 1)
            label_1 = Label(root10, text = "wrong rain value",font=('arial',16,'bold'),fg='red').grid(row = 0,column = 0)
            label_3 = Label(root10, text = "wrong pH value",font=('arial',16,'bold'),fg='red').grid(row = 2)
            predict()
            
        if(flag==0):
            predout()
            
    def predout():
        global sv,labelText,pred
        
        pred = sv.predict([[Entry_1.get(),Entry_2.get(),Entry_3.get()]])
        num=int(''.join(map(str,pred))) #or can use pred[0]
        data=croop[num]
        
        output.delete(0, END)
        output.insert(0,data)
       
        
        labelText = StringVar()
        labelText.set(data)
        
        
    def predout_prod():
        global pred,file1
        print(pred[0])
        p=pred[0]
        data1 = pd.read_csv(file1)
        le = preprocessing.LabelEncoder() #convert crops into numerical values

        Crop = le.fit_transform(data1.Crop) 

        testing1 = data1 

        testing1['Crop'] = Crop #replace crop with their respective numericals in the dataset
        predictors = testing1.drop(" Production",axis=1)
        target = testing1[" Production"]

        X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

        reg= RandomForestRegressor()
        reg.fit(X_train,Y_train)

        # prediction
        y_pred=reg.predict(X_test)
        y_pred
         
        score_dtr = r2_score(y_pred,Y_test)
        print("The r2 score achieved using Random Forest Regressor is: "+str(score_dtr))
        
        pred_1 = reg.predict([[float(Entry_1.get()),float(Entry_2.get()),float(Entry_3.get()),p]])
        

        output_8.delete(0, END)
        output_8.insert(0,str(abs(pred_1[0])))
     
        
     
        
    label_7 = Button(root10, text = 'crop',font=("Helvetica", 16),background="gray",command = acc_pH)
    label_7.grid(row=6,column=0)
    


    
    output = Entry(root10)
    output.grid(row=6,column=1)
    
    

    
B = Button(root, text = "Train and Test",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="gray",command=train_file)
B.grid(row=1,column=0)

B1 = Button(root, text = "Predict",height=1,padx=16,pady=16,bd=8,font=('arial',16,'bold'),width=10,bg="gray",command=predict)
B1.grid(row=1,column=4)

root.mainloop()
