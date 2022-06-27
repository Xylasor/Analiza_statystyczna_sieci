# TASS - PROJEKT 1
# AUTOR: MATEUSZ HRYCIOW 283365


import networkx as nx
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import powerlaw
import random


#1. Wczytanie grafu - "Siec stron www" oraz zbadanie rzedu oraz rozmiaru sieci
G = nx.read_edgelist('zadB_dane.txt', create_using=nx.MultiGraph)
print('Liczba wezlow i krawedzi pierwotnej sieci:')
print(G.number_of_nodes())
print(G.number_of_edges())

# Usuniecie petli i duplikatow krawedzi
G = nx.Graph(G)
print('Liczba wezlow i krawedzi po usunieciu duplikatow krawedzi:')
print(G.number_of_nodes())
print(G.number_of_edges())
G.remove_edges_from(nx.selfloop_edges(G))
print('Liczba wezlow i krawedzi po usunieciu petli i duplikatow:')
print(G.number_of_nodes())
print(G.number_of_edges())


#2. Wyodrebnienie najwiekszej skladowej spojnej
G_ss = max(nx.connected_components(G), key=len)
G_ss = G.subgraph(G_ss)
print('Liczba wezlow i krawedzi najwiekszej skladowej spojnej')
print(G_ss.number_of_nodes())
print(G_ss.number_of_edges())


#3. Wyznaczanie aproksymacji sredniej dlugosci sciezki
lengths = [100, 1000, 10000]
iters = 10
G_edges = G_ss.edges
for lens in (lengths):
    if lens == 10000:
        iters = 1
    connected_count = []
    paths_sum = 0
    for j in range(iters):
        print(j)
        G_len_edges = random.sample(G_ss.edges, int(lens/2))
        G_len = nx.Graph()
        G_len.add_edges_from(G_len_edges)
        while G_len.number_of_nodes() < lens:
            edge_add = random.sample(G_edges, int(lens/100))
            G_len.add_edges_from(edge_add)

        G_len_ss = max(nx.connected_components(G_len), key=len)
        connected_count.append(len(G_len_ss))
        G_len_ss = G_ss.subgraph(G_len_ss)
        shortest_path = nx.average_shortest_path_length(G_len_ss)
        paths_sum += shortest_path
    paths_avg = paths_sum/iters
    print("Rzedy sieci spojnych: ", connected_count, "dla liczby wezlow ", lens)
    print("Srednia dlugosc sciezki", paths_avg)


#4. Wyznaczanie liczby rdzeni o mozliwe najwiekszym rzedzie
vertex_degree = nx.core_number(G_ss)
vertex_degree_total = sorted(Counter(vertex_degree.values()).items())
print("Trzy najwyzsze rzedy to",vertex_degree_total[-1][0],vertex_degree_total[-2][0],vertex_degree_total[-3][0])

Core_1 = nx.k_core(G_ss, k = vertex_degree_total[-1][0])
print('Liczba wezlow i krawedzi k-rdzenia')
print(Core_1.number_of_nodes())
print(Core_1.number_of_edges())
Core_1_ss = max(nx.connected_components(Core_1), key=len)
Core_1_ss = G_ss.subgraph(Core_1_ss)
print('Liczba wezlow i krawedzi najwiekszej skladowej spojnej')
print(Core_1_ss.number_of_nodes())
print(Core_1_ss.number_of_edges())

Core_2 = nx.k_core(G_ss, k = vertex_degree_total[-2][0])
print('Liczba wezlow i krawedzi k-rdzenia')
print(Core_2.number_of_nodes())
print(Core_2.number_of_edges())
Core_2_ss = max(nx.connected_components(Core_2), key=len)
Core_2_ss = G_ss.subgraph(Core_2_ss)
print('Liczba wezlow i krawedzi najwiekszej skladowej spojnej')
print(Core_2_ss.number_of_nodes())
print(Core_2_ss.number_of_edges())

Core_3 = nx.k_core(G_ss, k = vertex_degree_total[-3][0])
print('Liczba wezlow i krawedzi k-rdzenia')
print(Core_3.number_of_nodes())
print(Core_3.number_of_edges())
Core_3_ss = max(nx.connected_components(Core_3), key=len)
Core_3_ss = G_ss.subgraph(Core_3_ss)
print('Liczba wezlow i krawedzi najwiekszej skladowej spojnej')
print(Core_3_ss.number_of_nodes())
print(Core_3_ss.number_of_edges())


#5. Wykreslanie rozkladu stopni wierzcholkow

degrees = G_ss.degree()
degrees = [deg[1] for deg in degrees]
degreeCount = Counter(degrees)
labels, values = zip(*degreeCount.items())

plt.bar(labels, values, color="b", width=0.7)
axes = plt.gca()
axes.set_xlim([0,40])
axes.set_ylim([0,200000])
plt.title("Histogram stopni wezlow")
plt.xlabel("Stopień wezla")
plt.ylabel("Liczba")
plt.show()

#6. Wyznaczanie wykladnika rozkladu potegowego

# Wykres liczby wezlow
degs = list(degreeCount.keys())
freqs = list(degreeCount.values())
freqs = [f for _,f in sorted(zip(degs,freqs))]
degs = sorted(degs)
fig = plt.figure()
ax = plt.gca()
ax.scatter(degs,freqs)
ax.set_yscale('log')
ax.set_xscale('log')
plt.title('Rozklad stopni wezlow')
plt.xlabel("Stopień wezla")
plt.ylabel("Liczba wezlow")
model = np.polyfit(np.log10(degs), np.log10(freqs), 1)
x_line = [min(degs), max(degs)]
y_line = [pow(10,model[1])*pow(x,model[0]) for x in x_line]
plt.plot(x_line, y_line, 'r')
plt.show()

# Wykres skumulowany
freqs_cum = []
for i in range(len(degs)):
    suma = 0
    for j in range (i, len(degs)):
        suma = suma + freqs[j]
    freqs_cum.append(suma)

fig = plt.figure()
ax = plt.gca()
ax.scatter(degs,freqs_cum)
ax.set_yscale('log')
ax.set_xscale('log')
plt.title('Rangowy rozklad stopni wezlow')
plt.xlabel("Stopień wezla")
plt.ylabel("Liczba wezlow o co najmniej tym stopniu")
model = np.polyfit(np.log10(degs), np.log10(freqs_cum), 1)
x_line = [min(degs), max(degs)]
y_line = [pow(10,model[1])*pow(x,model[0]) for x in x_line]
plt.plot(x_line, y_line, 'r')
plt.show()

przedzialy_N = 5
przedzialy = np.logspace(np.log10(min(degs)),np.log10(max(degs)),przedzialy_N+1)

fig = plt.figure()
ax = plt.gca()
ax.scatter(degs,freqs_cum)
ax.set_yscale('log')
ax.set_xscale('log')
plt.title('Rangowy rozklad stopni wezlow')
plt.xlabel("Stopień wezla")
plt.ylabel("Liczba wezlow o co najmniej tym stopniu")

fmin = 0
for i in range(1,przedzialy_N+1):

    deg_przedzial = [d for d in degs if d <= przedzialy[i]]
    fmax = degs.index(deg_przedzial[-1])
    freq_przedzial = freqs_cum[fmin:fmax+1]

    model = np.polyfit(np.log10(degs[fmin:fmax+1]), np.log10(freq_przedzial), 1)
    x_line = [degs[fmin], degs[fmax+1]]
    y_line = [pow(10,model[1])*pow(x,model[0]) for x in x_line]
    plt.plot(x_line, y_line, 'r')
    plt.axvline(degs[fmax+1], ls = "--")
    fmin = fmax + 1
    print("Nachylenie dla przedzialu ", i, " wynosi ", model[0])
plt.show()

#7. Wyzaczanie wykresu Hilla

NBINS = 50
bins = np.logspace(np.log10(min(degs)), np.log10(max(degs)), num = NBINS)
bcnt, bedge = np.histogram(np.array(degs), bins = bins)
alpha = np.zeros(len(bedge[:-2]))

for i in range(0, len(bedge)-2):
    fit = powerlaw.Fit(degs, xmin = bedge[i], discrete = True)
    alpha[i]=fit.alpha

plt.semilogx(bedge[:-2],alpha)
plt.title('Wykres Hilla')
plt.xlabel("Stopien wezla")
plt.ylabel("Alpha")
plt.show()