import streamlit as st
import numpy as np
import regex as re
import pandas as pd
import pygad
import random as r

width = 0
height = 0
roomMatrix = []
person_chance = 0.15

# FUNCIONES DE AYUDA GENERALES PARA EL SCRIPT


def prettyPrintArray(arr):
    # Wrapper del print de pandas, para imprimir arrays bonitos
    print(pd.DataFrame(arr))


def translateArrayTo2D(solution):
    # Convierte de un array 1D a 2D segun dimensiones de la habitacion
    global width, height
    newSol = []
    for i in range(height):
        newSol.append([])
        for j in range(width):
            newSol[i].append(2)

    # prettyPrintArray(newSol)

    for i in range(len(solution)):
        value = solution[i]

        jj = i % width
        ii = i // width

        newSol[ii][jj] = value

    return newSol


def translate2DToArray(arr):
    # Convierte un array 2D a 1D
    global width, height

    newArr = []
    for i in range(width*height):
        newArr.append(0)

    for i in range(height):
        for j in range(width):
            k = j + height * i
            # print(k)
            newArr[k] = arr[i][j]

    return newArr


def leer_habitacion(file_name):
    global roomMatrix, width, height
    # Genera la matriz que representa la habitación desde un archivo *.txt
    file = open(file_name, 'r')

    initLineRecognized = False
    roomMatrix = []
    for line in file.readlines():
        line = line.replace('\n', '')

        if re.match('^[-]*$', line):
            print(f'Linea de {"salida" if initLineRecognized else "entrada"} reconocida')
            initLineRecognized = True
            continue
        elif re.match('^[0-9]*,[0-9]*$', line):
            width, height = line.split(',')
            width, height = int(width), int(height)
            print(f"Ancho y Alto de la habitacion encontrado (w: {width}, h: {height})")
            continue
        else:
            rowItem = []
            for c in line:
                rowItem.append(c)
            roomMatrix.append(rowItem)

    print(f"finalmente: Dimensiones(w: {width}, h: {height})")
    print(f"habitacion:")
    prettyPrintArray(roomMatrix)


def generateRoomMarkup(roomMatrix, peopleMatrix):
    cssString = """
    .room {
      display: flex;
      flex-direction: column;
    }
    .row {
      display: flex;
    }
    .cell {
      width: 20px;
      height: 20px;
      border: 1px solid black;
      display: flex;
      justify-content: center;
      align-items: center;
    }
    .free {
      background-color: white;
    }
    .wall {
      background-color: black;
    }
    .door {
      background-color: orange;
    }
    .table {
      background-color: orange;
    }
    .person{
      background-color: #BFFF00;
    }
    """
    htmlWrapperInit = f"<html><head></head><body><style>{cssString}</style>"
    htmlWrapperEnd = '</body></html>'

    roomWrapperInit = "<div class='room'>"
    roomWrapperEnd = "</div>"

    output = ""

    # Verifico que ambos arrays son de las mismas dimensiones
    assert len(roomMatrix) == len(peopleMatrix)
    assert len(roomMatrix[0]) == len(peopleMatrix[0])

    for i in range(len(roomMatrix)):
        output += "<div class='row'>"
        for j in range(len(roomMatrix[0])):
            roomCell = roomMatrix[i][j]
            cellContent = ""
            if roomCell == "X":
                cellTypeClass = "free"
            elif roomCell == " ":
                cellTypeClass = "wall"
            elif roomCell == "P":
                cellTypeClass = "door"
                cellContent = "P"
            elif roomCell == "M":
                cellTypeClass = "table"
                cellContent = "M"

            cellContentClass = "person" if peopleMatrix[i][j] == 1 else ""

            output += f"<div class='{cellTypeClass} {cellContentClass} cell'>{cellContent}</div>"
        output += "</div>"

    return f"{htmlWrapperInit}{roomWrapperInit}{output}{roomWrapperEnd}{htmlWrapperEnd}"


# def displayRoom(roomMatrix, peopleMatrix):
#    return display(HTML(generateRoomMarkup(roomMatrix, peopleMatrix)))


# Función para generar nuevas soluciones y poblacion
def generateSolution():
    global width, height, roomMatrix
    generatedSolution = []

    for i in range(height):
        generatedSolution.append([])
        for j in range(width):
            rand = r.random()

            cell = roomMatrix[i][j]
            isValidCell = cell == "X"

            isPerson = rand < person_chance and isValidCell

            generatedSolution[i].append(1 if isPerson else 0)

    return generatedSolution


def generatePopulation(roomMatrix, n):
    population = []
    for i in range(n):
        child = generateSolution()
        child = translate2DToArray(child)
        population.append(child)
    # print(population)
    return population


def fitnessFunction(solution, index):
    global width, height, roomMatrix

    if len(solution) == width * height:
        solution = translateArrayTo2D(solution)

    fitness = 0
    peopleOnInvalidCells = 0
    peopleCorrectlyPlaced = 0
    peopleNotDistancing = 0
    invalidTableArrangements = 0

    for i in range(height):
        for j in range(width):
            solCell = solution[i][j]
            roomCell = roomMatrix[i][j]

            # Si la celda actual contiene a una persona
            if solCell == 1:
                # Si se encuentra en una celda invalida, rip
                if roomCell == " " or roomCell == "P" or roomCell == "M":
                    #fitness -= 100
                    peopleOnInvalidCells += 1
                    continue
                    # return 0

                # Verifico si la persona no tiene a otra alrededor de ella
                isValid = checkSorroundingCellsForPeople(solution, i, j)
                # Si esta distanciada, fit +1
                if isValid:
                    #fitness += 1
                    peopleCorrectlyPlaced += 1
                    continue
                # Si no, rip
                else:
                    #print(f"Invalid position found around cell {i}, {j}")
                    #fitness -= 10
                    peopleNotDistancing += 1
                    continue
                    # return 0

            # Si la celda es una mesa, verificar las reglas de una mesa
            if roomCell == "M":
                isValid = checkForTableRules(solution, i, j)

                if isValid:
                    continue
                else:
                    #print(f"Invalid table arrangement found around cell {i}, {j}")
                    #fitness -= 100
                    invalidTableArrangements += 1
                    continue
                    # return 0

    # print(fitness)
    fitness += 2**peopleCorrectlyPlaced
    #fitness -= 10*peopleOnInvalidCells
    #fitness -= 2*peopleNotDistancing
    fitness -= 5*invalidTableArrangements
    if index == -1:
        print(f"fitness: {fitness}, peopleOnInvalidCells: {peopleOnInvalidCells}, peopleCorrectlyPlaced: {peopleCorrectlyPlaced}, peopleNotDistancing: {peopleNotDistancing}, invalidTableArrangements: {invalidTableArrangements}")

    return fitness


def checkForTableRules(solution, i, j):
    global roomMatrix
    numPeopleAllowedAroundTables = 1

    width = len(solution)
    height = len(solution[0])

    leftCell = [i, j-1]
    rightCell = [i, j+1]
    topCell = [i-1, j]
    botCell = [i+1, j]
    cells = [leftCell, rightCell, topCell, botCell]

    peopleCount = 0
    for cell in cells:

        if cell[0] < 0 or cell[0] >= height or cell[1] < 0 or cell[1] >= width:
            continue

        if roomMatrix[cell[0]][cell[1]] != "X":
            continue

        cellValue = solution[cell[0]][cell[1]]
        isOccupied = cellValue == 1
        if isOccupied:
            peopleCount += 1

    return peopleCount <= numPeopleAllowedAroundTables


def checkSorroundingCellsForPeople(solution, i, j):
    global width, height, roomMatrix

    #print("\nchecking around cell: ",i,j)

    for ii in range(i-1, i+2):
        for jj in range(j-1, j+2):
            # Excluding checked cell itself
            if ii == i and jj == j:
                #print("excluding self")
                continue
            # Excluding coordinates out of room boundaries
            if ii < 0 or ii >= height or jj < 0 or jj >= width:
                #print("out of bounds at: ", ii,jj)
                continue

            if roomMatrix[ii][jj] != "X":
                continue

            if solution[ii][jj] == 1:
                #print("Invalid position at: ",ii,jj)
                return False
    #print("completed check around cell: ",i,j)
    return True

# Determinar fenotipo


def generatePhenotype(solution):
    global width, height, roomMatrix

    peopleCounter = 0

    if len(solution) == width * height:
        solution = translateArrayTo2D(solution)

    for i in range(height):
        for j in range(width):
            solCell = solution[i][j]
            roomCell = roomMatrix[i][j]

            # Si la celda actual contiene a una persona
            if solCell == 1:
                # Si se encuentra en una celda invalida, rip
                if roomCell == " " or roomCell == "P" or roomCell == "M":
                    solution[i][j] = 0
                    continue

                # Verifico si la persona no tiene a otra alrededor de ella
                isValid = checkSorroundingCellsForPeople(solution, i, j)
                if isValid:
                    peopleCounter += 1
                    continue
                # Si no, rip
                else:
                    solution[i][j] = 0
                    continue

    return solution, peopleCounter


last_fitness = 0
max_people_at = [0, 0, []]


def callback_generation(ga_instance):
    global last_fitness, max_people_at, roomMatrix

    generation = ga_instance.generations_completed
    best_fitness = ga_instance.best_solution()[1]
    fitness_delta = ga_instance.best_solution()[1] - last_fitness

    print("Generation = {generation}".format(generation=ga_instance.generations_completed),
          ", Fitness Best = {fitness:.3f}".format(fitness=ga_instance.best_solution()[1]),
          ", Change = {change:.3f}".format(change=ga_instance.best_solution()[1] - last_fitness))

    fitnessFunction(ga_instance.best_solution()[0], -1)
    phenotype, peopleCounter = generatePhenotype(ga_instance.best_solution()[0])
    print(f"# Personas: {peopleCounter}")
    if peopleCounter > max_people_at[1]:
        max_people_at = [generation, peopleCounter, phenotype]
    #displayRoom(roomMatrix, phenotype)

    last_fitness = ga_instance.best_solution()[1]

    if ga_instance.generations_completed % 10 == 0:
        # ga_instance.plot_fitness()
        pass

    textHolder.markdown(f"""
    <p>Generation: {generation}, Best fitness: {best_fitness}, Change: {fitness_delta}</p>
    <p>Best solution so far: {max_people_at[1]} people in gen {max_people_at[0]}</p>
    """, unsafe_allow_html=True)
    roomHolder.markdown(f"""
    Current gen best room has {peopleCounter} inhabitants and looks like: \n{generateRoomMarkup(roomMatrix, phenotype)}
    Best solution so far is in gen {max_people_at[0]}. The room has {max_people_at[1]} inhabitants and looks like: \n{generateRoomMarkup(roomMatrix, max_people_at[2])}
    """, unsafe_allow_html=True)


def run_algorithm():
    num_generations = 1000  # Number of generations.
    population_size = 100
    num_parents_mating = 7  # Number of solutions to be selected as parents in the mating pool.
    parent_selection_type = "tournament"  # Type of parent selection.
    keep_parents = 7  # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.
    crossover_type = "single_point"  # Type of the crossover operator.
    # Parameters of the mutation operation.
    mutation_type = "random"  # Type of the mutation operator.
    mutation_percent_genes = 35  # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists or when mutation_type is None.

    initial_population = generatePopulation(roomMatrix, population_size)
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitnessFunction,
                           parent_selection_type=parent_selection_type,
                           gene_space=[0, 1],
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           on_generation=callback_generation,
                           initial_population=initial_population
                           )
    ga_instance.run()

# FUNCIONES PERTINENTES A LA GENERACION DE LA PAGINA WEB CON STREAMLIT


def generar_pagina():
    st.set_page_config(
        page_title="IA - Clasificador de frutas",
        page_icon="🍎"
    )

    st.write("# Genetic Algorithm - COVID-19 Social Distancing.")
    st.markdown('by Víctor Matamala & Matías Rivera', unsafe_allow_html=True)

    with st.expander("🧙 Click here to learn more about this project 🔮"):
        st.markdown("""
            <p>A Genetic Algorithm (GA) may be attributed as a method for optimizing the search tool for solutions to problems hard to model algorithmically, 
            commonly used for optimization and search problems. This methodology is inspired on our understanding of the principles of natural selection 
            with analogous abstractions for chromosome generation, how fit each solution is to the problem, natural parent selection and gene crossover 
            and mutation for the offspring. The problem at hand focuses on how many people can safely fit on a room, following an abstraction of the 
            commonly used guidelines for social distancing.</p>
        """, unsafe_allow_html=True)

    st.button("Start simulation", on_click=iniciar_script)

    global textHolder, roomHolder
    textHolder = st.empty()
    roomHolder = st.empty()


def iniciar_script():
    run_algorithm()
    pass


def main():
    generar_pagina()

    leer_habitacion('input.txt')
    pass


if __name__ == "__main__":
    main()
