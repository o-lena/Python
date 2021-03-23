import PySimpleGUI as sg
import openpyxl

roster = openpyxl.load_workbook(r"C:\Users\Lena\Desktop\python\merc.xlsx")
sheet = roster.active  

sg.theme('DefaultNoMoreNagging')
#sg.theme('TanBlue')
#sg.theme('Material2')
layout =  [[sg.Text(pad=(150,30), font=("Arial", 20, 'bold'), size=(15,1))],
          [sg.Text("model:", font=("Arial", 15, 'bold'), pad=(20,0), size=(12,1)), sg.InputText(key='model_1', font="Arial 20", background_color="#F7F9F9", size=(10,1)),sg.Button("Enter",font="Arial 10", pad=(10,0))],
          [sg.Text("year", font=("Arial", 15, 'bold'), pad=(20,30), size=(12,1)), sg.Multiline(key='year_1', font="Arial 20", pad=(0,5), background_color="#F7F9F9", size=(17,1), do_not_clear=False)]]

window = sg.Window("Working Code", layout)

while True:
    event,values = window.read()
    if event == "Cancel" or event == sg.WIN_CLOSED:
        break
    elif event == "Enter":        
        for row in sheet.rows:
            if  int(values['model_1']) == row[2].value:
                print("year:{} {}".format(row[0].value,row[1].value))        
                window['year_1'].update(f"{row[0].value} {row[1].value}")
                break
        else:
            sg.popup_error("Record not found", font="Arial 10")

window.close()