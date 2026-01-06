import pygame
import math
from queue import PriorityQueue

ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Algoritmo A*")

# --- PALETA DE COLORES ---
BLANCO = (255, 255, 255)       # Fondo (Sin cambios)
NEGRO = (0, 0, 0)              # Pared (Sin cambios)

# Nuevos colores
TURQUESA = (64, 224, 208)      # Inicio
AZUL_REAL = (65, 105, 225)     # Fin
DORADO = (255, 215, 0)         # Camino final
SALMON = (250, 128, 114)       # Nodos ya visitados (Closed set)
MENTA = (152, 251, 152)        # Nodos en espera (Open set)
GRIS_CLARO = (220, 220, 220)   # Líneas de la cuadrícul

orden_revision = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0]]


class Nodo:
    def __init__(self, fila, col, ancho, total_filas, n_nodo, n_padre, g_costo=float('inf'), h_heuristica=float('inf')):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas
        self.n_nodo = n_nodo
        self.n_padre = n_padre
        self.g_costo = g_costo
        self.h_heuristica = h_heuristica
    
    def get_datos_nodo(self):
        f_total = self.g_costo + self.h_heuristica
        return self.n_nodo, self.n_padre, self.g_costo, self.h_heuristica, f_total
    
    def set_n_nodo(self, n_nodo):
        self.n_nodo = n_nodo

    def set_n_padre(self, n_padre):
        self.n_padre = n_padre

    def set_g_costo(self, g_costo):
        self.g_costo = g_costo

    def set_h_heuristica(self, h_heuristica):
        self.h_heuristica = h_heuristica

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == TURQUESA

    def es_fin(self):
        return self.color == AZUL_REAL

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = TURQUESA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = AZUL_REAL

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas, None, None)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        # Usamos GRIS_CLARO para las líneas
        pygame.draw.line(ventana, GRIS_CLARO, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        pygame.draw.line(ventana, GRIS_CLARO, (i * ancho_nodo, 0), (i * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(BLANCO)
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    y, x = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def h(p1_pos, p2_pos):
    fila1, col1 = p1_pos
    fila2, col2 = p2_pos
    dx = abs(fila1 - fila2)
    dy = abs(col1 - col2)
    costo_diagonal = math.sqrt(2)
    costo_ortogonal = 1
    return (costo_ortogonal * (max(dx, dy) - min(dx, dy)) + costo_diagonal * min(dx, dy))

def dentro_de_limite(max_limite, num):
    min_limite = 0
    if num >= min_limite and num < max_limite:
        return True
    return False

def no_repetidos(lista, nuevo_dato):
    longitud = len(lista)
    for i in range(longitud):
        if lista[i] == nuevo_dato:
            return True
    return False

def se_pueden_usar(grid, nodo_actual):
    fila_actual = nodo_actual.fila
    col_actual = nodo_actual.col
    max_filas = len(grid)
    lista_usables = []

    for offset in orden_revision: 
        offset_fila = offset[0]
        offset_col = offset[1]
        nueva_fila = fila_actual + offset_fila
        nueva_col = col_actual + offset_col
        
        if dentro_de_limite(max_filas, nueva_fila) and dentro_de_limite(max_filas, nueva_col):
            vecino = grid[nueva_fila][nueva_col]
            if not vecino.es_pared():
                es_diagonal = (offset_fila != 0 and offset_col != 0)
                if es_diagonal: 
                    fila_flanco_1 = fila_actual
                    col_flanco_1 = col_actual + offset_col
                    fila_flanco_2 = fila_actual + offset_fila
                    col_flanco_2 = col_actual
                    pared_en_flanco_1 = grid[fila_flanco_1][col_flanco_1].es_pared()
                    pared_en_flanco_2 = grid[fila_flanco_2][col_flanco_2].es_pared()
                    if not pared_en_flanco_1 or not pared_en_flanco_2:
                        lista_usables.append(vecino)
                else:
                    lista_usables.append(vecino)
    return lista_usables

def reconstruir_camino(ventana, grid, filas, ancho, nodo_actual):
    while nodo_actual.n_padre is not None:
        nodo_actual = nodo_actual.n_padre
        if not nodo_actual.es_inicio():
            nodo_actual.color = DORADO # Color del camino final

def iniciar_algoritmo(ventana, grid, filas, ancho, inicio, fin):
    contador = 0
    open_set = PriorityQueue()
    
    open_set.put((0, contador, inicio))
    
    open_set_hash = {inicio}
    
    inicio.set_g_costo(0)
    h_score_inicio = h(inicio.get_pos(), fin.get_pos())
    inicio.set_h_heuristica(h_score_inicio)

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        nodo_actual = open_set.get()[2] 
        open_set_hash.remove(nodo_actual)

        if nodo_actual == fin:
            reconstruir_camino(ventana, grid, filas, ancho, fin)
            fin.hacer_fin()
            inicio.hacer_inicio()
            return True

        vecinos = se_pueden_usar(grid, nodo_actual)
        
        for vecino in vecinos:
            costo = 1.0
            if abs(vecino.fila - nodo_actual.fila) == 1 and abs(vecino.col - nodo_actual.col) == 1:
                costo = math.sqrt(2)
            
            g_tentativo = nodo_actual.g_costo + costo

            if g_tentativo < vecino.g_costo:
                vecino.set_n_padre(nodo_actual)
                vecino.set_g_costo(g_tentativo)
                
                h_score_vecino = h(vecino.get_pos(), fin.get_pos()) * 1.0001
                vecino.set_h_heuristica(h_score_vecino)
                
                f_total = g_tentativo + h_score_vecino

                if vecino not in open_set_hash:
                    contador += 1
                    open_set.put((f_total, contador, vecino))
                    open_set_hash.add(vecino)
                    vecino.color = MENTA # Color del Open Set (Por visitar)

        dibujar(ventana, grid, filas, ancho)

        if nodo_actual != inicio:
            nodo_actual.color = SALMON # Color del Closed Set (Visitados)

    return False

def main(ventana, ancho):
    FILAS = 11
    grid = crear_grid(FILAS, ancho)

    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False
            
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                
                if 0 <= fila < FILAS and 0 <= col < FILAS:
                    nodo = grid[fila][col]
                    if not inicio and nodo != fin:
                        inicio = nodo
                        inicio.hacer_inicio()
                    elif not fin and nodo != inicio:
                        fin = nodo
                        fin.hacer_fin()
                    elif nodo != fin and nodo != inicio:
                        nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                
                if 0 <= fila < FILAS and 0 <= col < FILAS:
                    nodo = grid[fila][col]
                    nodo.restablecer()
                    if nodo == inicio:
                        inicio = None
                    elif nodo == fin:
                        fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            # Reseteamos si tiene alguno de los nuevos colores
                            if nodo.color == MENTA or nodo.color == SALMON or nodo.color == DORADO:
                                nodo.restablecer()
                            nodo.set_n_nodo(None)
                            nodo.set_n_padre(None)
                            nodo.set_g_costo(float('inf'))
                            nodo.set_h_heuristica(float('inf'))
                    
                    encontrado = iniciar_algoritmo(ventana, grid, FILAS, ancho, inicio, fin)
                    
                    if not encontrado:
                        print("¡ALGORITMO TERMINADO! No se encontró un camino.")

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)