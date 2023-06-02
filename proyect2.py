import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn import tree # Importa la función del arbol
import csv
import time


def modelo_knn(X_heart, Y_heart, vecinos):
    print("vecinos =", vecinos)
    X_train, X_test, y_train, y_test = train_test_split(X_heart, Y_heart, test_size=0.20, random_state=None) # Se divide la data en un 80-20

    algoritmo = KNeighborsClassifier(n_neighbors=vecinos)
    algoritmo.fit(X_train, y_train)
    y_pred = algoritmo.predict(X_test) 
    return y_test, y_pred

    
def modelo_dt(X_heart, Y_heart, ramas):
    '''
    Modelo decission Tree
    divide el set en 80-20
    entrea y predice, la función metricas muestra después 
    los scores de evaluación
    '''
    X_train, X_test, y_train, y_test = train_test_split(X_heart, Y_heart, test_size=0.20, random_state=None) # Se divide la data en un 80-20
    algoritmo = DecisionTreeClassifier(max_depth=ramas)
    # entrenar
    algoritmo.fit(X_train, y_train)
    #print("test")
    #print(y_test)

    # realizar prediccioón
    y_pred = algoritmo.predict(X_test) 
    #print(y_pred)
    return y_test, y_pred, algoritmo


def modelo_nb(X_heart, Y_heart):
    X_train, X_test, y_train, y_test = train_test_split(X_heart, Y_heart, test_size=0.20, random_state=None) # Se divide la data en un 80-20
    algoritmo = GaussianNB()
    algoritmo.fit(X_train,y_train)
    y_pred = algoritmo.predict(X_test)
    return y_test, y_pred

def modelo_rf(X_heart, Y_heart, estimador):
    X_train, X_test, y_train, y_test = train_test_split(X_heart, Y_heart, test_size=0.20, random_state=None) # Se divide la data en un 80-20
    algoritmo = RandomForestClassifier(n_estimators=estimador)
    algoritmo.fit(X_train, y_train)
    y_pred = algoritmo.predict(X_test)
    return y_test, y_pred
    

def metricas(y_test, y_pred, name):

    print("Métricas considerando la clase 0 como positiva")
    matriz = confusion_matrix(y_test, y_pred)
   
    print(matriz)
    #sns.heatmap(matriz, annot=True, cmap="Blues")
    #plt.title("Matriz de Confusión")
    #plt.xlabel("Predicciones")
    #plt.ylabel("Valores reales")
    #plt.savefig(name)

    print (classification_report(y_test, y_pred))

    # calculo precision
    precision = precision_score(y_test, y_pred, pos_label=0)
    #precision = tn / (tn+fn)
    print('Precisión:', precision)


    # calculo accyracy eficiencia global
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)

    # calculo recall tasa verdaderos positivos
    recall = recall_score(y_test, y_pred, pos_label=0)
    print("recall", recall)
    return precision, accuracy, recall


def menu():
    # leer el archivo
    datos = pd.read_csv('https://raw.githubusercontent.com/valenoe/Proyecto_2_mineria_de_datos/main/set_normalizado_grupo_06.csv')

    # se elimina el id que genera pandas
    datos.drop(columns=["id"], inplace=True)
    

    # separa las columnas con atributos de la que contienen la clase

    nombres_columnas = datos.columns.values
    #print("columnas: ", nombres_columnas)
    nombres_columnas = nombres_columnas[:-1]
    #print("columnas sin la clase: ", nombres_columnas)
    numeros_clase1 = 0
    numeros_clase0 = 0
    for i in datos["output"]:
        if i == 1:
            numeros_clase1 += 1
        if i == 0:
            numeros_clase0 += 1

    print("hay ", numeros_clase1, "de la clase 1 (+ probb de ataque) y ", numeros_clase0, "de la clase 0 (-prob de ataque)")
            

    #atributo_columnas = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    X_heart = datos[nombres_columnas]
    Y_heart = datos.output

    
    
    labels = ["Posible enfermo", "Posible sano"]
    count_classes = pd.value_counts(datos['output'], sort = True)
    count_classes.plot(kind = 'bar', rot=0)
    plt.xticks(range(2), labels)
    plt.title("Frequency by observation number")
    plt.xlabel("Class")
    plt.ylabel("Number of Observations")
    plt.savefig("cantidades.png")

    print("Esperando...")
    print("Parsando a modelo DT")
    time.sleep(2)
    
    '''
    para el modelo decission Tree
    primero se muestra el modelo y sus metricas,
    luego se hace la validacióon cruzada 5 veces
    '''
    
    name_metricas = ["precision", "accuracy", "recall"]
    metricas_lista_dt_np = []
    y_test_dt, y_pred_dt, algoritmo = modelo_dt(X_heart, Y_heart, None)
    metricas_lista_dt_np = metricas(y_test_dt, y_pred_dt, "cm_dt.png")
    plt.figure(figsize=(30,15))  # Tamaño de la ventana
    tree.plot_tree(algoritmo, class_names=True, fontsize=13) # Crea el árbol mostrando la clase y la "fontsize" que es el tamaño de la ventana
    plt.savefig("dt_sinPoda.png")

    scores_dt = []
    for i in range(5):
        y_test_dt_v, y_pred_dt_v, algoritmo = modelo_dt(X_heart, Y_heart, None)
        scores_dt.append(accuracy_score(y_test_dt_v, y_pred_dt_v,))

    print("accuracy dt:", scores_dt)
    print(name_metricas)
    print(metricas_lista_dt_np)

    
    # ahora para el arbol podado
    y_test_dt_poda, y_pred_dt_poda, algoritmo_poda = modelo_dt(X_heart, Y_heart, 4)
    metricas_lista_dt_p =[]
    metricas_lista_dt_p= metricas(y_test_dt_poda, y_pred_dt_poda, "cm_dt_poda.png")
    plt.figure(figsize=(30,15))  # Tamaño de la ventana
    tree.plot_tree(algoritmo_poda, class_names=True, fontsize=13) # Crea el árbol mostrando la clase y la "fontsize" que es el tamaño de la ventana
    plt.savefig("dt_conPoda.png")

    scores_dt_poda = []
    for i in range(5):
        y_test_dt_v_p, y_pred_dt_v_p, algoritmo = modelo_dt(X_heart, Y_heart, 4)
        scores_dt_poda.append(accuracy_score(y_test_dt_v_p, y_pred_dt_v_p,))

    print("accuracy dt con poda de 4:", scores_dt_poda)

    
    print("Esperando...")
    time.sleep(1)
    print("Parsando a modelo KNN")
    time.sleep(4)



    metricas_knn_1_10 = []
    scores_knn_lista = []
    for i in range(1,11):
        print(i)
        y_test_knn, y_pred_knn= modelo_knn(X_heart, Y_heart, i)
        metricas_lista_knn =[]
        name = f"matriz_knn_vecino_{i}.png"
        metricas_lista_knn= metricas(y_test_knn, y_pred_knn, name)
        metricas_knn_1_10.append(metricas_lista_knn)
        scores_knn = []
        for j in range(5):
            y_test_knn_v, y_pred_knn_v= modelo_knn(X_heart, Y_heart, i)
            scores_knn.append(accuracy_score(y_test_knn_v, y_pred_knn_v,))
        scores_knn_lista.append(scores_knn)
    

    
    print("Esperando...")
    time.sleep(1)
    print("Parsando a modelo NB")
    time.sleep(4)
    
    metricas_lista_naba = []
    y_test_nb, y_pred_nb = modelo_nb(X_heart, Y_heart)
    metricas_lista_naba = metricas(y_test_nb, y_pred_nb, "cm_nb.png")
    
    scores_naba = []
    for i in range(5):
        y_test_nb_v, y_pred_nb_v = modelo_nb(X_heart, Y_heart)
        scores_naba.append(accuracy_score(y_test_nb_v, y_pred_nb_v,))

    print("accuracy nb:", scores_naba)


    print("Esperando...")
    time.sleep(1)
    print("Parsando a modelo RF")
    time.sleep(4)
    
    metricas_lista_rf = []
    scores_rf_lista = []
    lista_para_rf = [10,20,50, 100, 500]
    for x in lista_para_rf:
        y_test_rf, y_pred_rf = modelo_rf(X_heart, Y_heart, x)
        print("\nCantidad de árboles: ", x)
        metricas_lista_rf.append(metricas(y_test_rf, y_pred_rf, "cm_rf.png"))
        scores_rf = []
        for i in range(5):
            y_test_rf_v, y_pred_rf_v = modelo_rf(X_heart, Y_heart, x)
            print("\nCantidad de árboles: ", x)
            scores_rf.append(accuracy_score(y_test_rf_v, y_pred_rf_v,))
        scores_rf_lista.append(scores_rf)

    print()
    for i in metricas_lista_rf:
        print(i)

    print()
    for i in scores_rf_lista:
        print(i)

    
    
        
            

    
    # Definir el nombre del archivo CSV
    nombre_archivo = 'metricas.csv'

    # Crear y abrir el archivo CSV en modo de escritura
    with open(nombre_archivo, mode='w', newline='') as archivo:
        writer = csv.writer(archivo)
        writer.writerow(["N° class 0", numeros_clase0])
        writer.writerow(["N° class 1", numeros_clase1])
        writer.writerow(["Total:", numeros_clase0+numeros_clase1])


        writer.writerow("")
        writer.writerow(["Árbol de decisión"])
        writer.writerow("")
        writer.writerow(["Metricas"])
        # Escribir el encabezado del archivo
        writer.writerow(['métricas', 'árbol no podado', 'árbol podado'])

        # Escribir las métricas en cada fila
        for i in range(len(name_metricas)):
            writer.writerow([ name_metricas[i], metricas_lista_dt_np[i], metricas_lista_dt_p[i]])

        # accuracy para validación cruzada DT
        writer.writerow("")
        writer.writerow(["Accuracy por validación cruzada"])
        writer.writerow(['n', 'accuracy no podado', 'accuracy podado'])

        for i in range(5):
            writer.writerow([i+1, scores_dt[i], scores_dt_poda[i]])
        
        writer.writerow("")
        writer.writerow(["vecinos Cercanos"])
        writer.writerow("")
        writer.writerow("")
        writer.writerow(["Accuracy por validación cruzada"])
        # mtricas para KNN
        writer.writerow(['métricas', '1 vecino', '2 vecinos', '3 vecinos', '4 vecinos', '5 vecinos', '6 vecinos', '7 vecinos', '8 vecinos', '9 vecinos', '10 vecinos'])

        # Escribir las métricas en cada fila
        for i in range(len(name_metricas)):
            writer.writerow([ name_metricas[i],metricas_knn_1_10[0][i], metricas_knn_1_10[1][i], 
                             metricas_knn_1_10[2][i], metricas_knn_1_10[3][i],metricas_knn_1_10[4][i], 
                             metricas_knn_1_10[5][i],metricas_knn_1_10[6][i], metricas_knn_1_10[7][i],
                             metricas_knn_1_10[8][i], metricas_knn_1_10[9][i]])


        # accuracy para validación cruzada KNN
        writer.writerow("")
        writer.writerow(["Accuracy por validación cruzada"])

        writer.writerow(['n', '1 vecino', '2 vecinos', '3 vecinos', '4 vecinos', '5 vecinos', '6 vecinos', '7 vecinos', '8 vecinos', '9 vecinos', '10 vecinos'])

        for i in range(5): # nota: todos son 5
            writer.writerow([i+1, scores_knn_lista[0][i], scores_knn_lista[1][i], 
                             scores_knn_lista[2][i], scores_knn_lista[3][i],scores_knn_lista[4][i], 
                             scores_knn_lista[5][i],scores_knn_lista[6][i], scores_knn_lista[7][i],
                             scores_knn_lista[8][i], scores_knn_lista[9][i]])

        writer.writerow("")
        writer.writerow(["Naive Bayes"])
        writer.writerow("")
        writer.writerow(["Metricas"])
        writer.writerow(['métricas', 'modelo'])

        # Escribir las métricas en cada fila
        for i in range(len(name_metricas)):
            writer.writerow([ name_metricas[i], metricas_lista_naba[i]])

        writer.writerow("")
        writer.writerow(["Accuracy por validación cruzada"])
        writer.writerow(['n', 'modelo'])

        for i in range(5):
            writer.writerow([i+1, scores_naba[i]])

        writer.writerow("")
        writer.writerow(["Random Forest"])
        writer.writerow("")
        writer.writerow(["Metricas"])
        writer.writerow(['métricas', '10', '20', '50', '100', '500'])
        for i in range(len(name_metricas)):
            writer.writerow([name_metricas[i], metricas_lista_rf[0][i], metricas_lista_rf[1][i], metricas_lista_rf[2][i],
                             metricas_lista_rf[3][i], metricas_lista_rf[4][i]])
       
        writer.writerow("")
        writer.writerow(["Metricas"])
        writer.writerow(['n', '10', '20', '50', '100', '500'])
        for i in range(5):
            writer.writerow([i+1, scores_rf_lista[0][i], scores_rf_lista[1][i], scores_rf_lista[2][i],
                             scores_rf_lista[3][i], scores_rf_lista[4][i]])



    print("Las métricas se han guardado en el archivo", nombre_archivo)
    

if __name__ == '__main__':
    menu()
