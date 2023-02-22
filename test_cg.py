import scipy
import numpy as np
import inspect
import matplotlib.pyplot as plt

r_values = []
r_values_rate = []
r_values_cond = []
r_values_cond_rate = []
k_values = []
k_values_cond = []

iteration = 0

def callback(xk):
    frame = inspect.currentframe().f_back
    r = frame.f_locals['resid']     #gibt Residuum an
    r_values[iteration].append(r)
    k = frame.f_locals['olditer']   #gibt Anzahl Iterationen an
    rate = (r/r_values[iteration][0])**(1/k) #berechnet Änderungsrate
    r_values_rate[iteration].append(rate)  #fügt Änd.rate an entspr. Vektor (zum plotten)
    k_values[iteration].append(k)
    print('k=', k, '\t|r|={:10.9f}'.format(r), 'rate=', rate)


def callbackCond(xk):
    frame = inspect.currentframe().f_back
    r = frame.f_locals['resid']  # gibt Residual an
    r_values_cond[iteration].append(r)
    k = frame.f_locals['olditer']  # gibt Anzahl Iterationen an
    rate = (r / r_values_cond[iteration][0] ) ** (1 / k)
    r_values_cond_rate[iteration].append(rate)
    k_values_cond[iteration].append(k)
    print('k_cond=', k, '\t|r_cond|={:10.9f}'.format(r), 'rate=', rate)

n = np.zeros(shape=5, dtype=int)

#pro l ein Vektor, eine Matrix und je ein Aufruf cg

for l in range (0, 5):
    n[l] = 100*(2**l)
    b = np.zeros(shape=n[l])
    A = scipy.sparse.diags([-1, 0, -1], [-1, 0, 1], shape=(n[l], n[l])).toarray()
    D = np.zeros(shape=(n[l], n[l]))
    iteration = l
    for i in range (0, n[l]):
        b[i] = np.sin(i**2)

    for i in range (0, n[l]):
        A[i][i] = i+1
        D[i][i] = i+1

    B = np.linalg.inv(D)

    x0 = np.zeros(shape=n[l])
    r_values.append([])
    r_values_rate.append([])
    r_values_cond.append([])
    r_values_cond_rate.append([])
    k_values.append([])
    k_values_cond.append([])
    scipy.sparse.linalg.cg(A, b, x0, tol=10**-5, maxiter=None, M=None, callback = callback)    #M ist der Vorkonditionierer
    scipy.sparse.linalg.cg(A, b, x0, tol=10**-5, maxiter=None, M=B, callback = callbackCond)    #M ist der D^-1


if __name__ == "__main__":

    r_means = []
    r_means_cond = []

    #maximale Iterationen pro Dimension
    k_values_max = [k_values[0][-1],k_values[1][-1],k_values[2][-1],k_values[3][-1], k_values[4][-1]]
    k_values_cond_max = [k_values_cond[0][-1],k_values_cond[1][-1],k_values_cond[2][-1],k_values_cond[3][-1],k_values_cond[4][-1]]

    #Berechnung Mittelwerte der Änderungsraten
    for r_vec in r_values_rate:
        r_means.append(np.mean(r_vec))
    for r_vec in r_values_cond_rate:
        r_means_cond.append(np.mean(r_vec))

    #plot der Änderungsraten
    plt.plot(n, r_means, label='ohne Vorkonditionierung')
    plt.plot(n, r_means_cond, label='mit Vorkonditionierung')
    plt.title('Durschnittliche Änderungsrate pro Dimension')
    plt.legend()
    plt.show()

    #plot der Anzahl Iteration
    plt.plot(n, k_values_max, label='ohne Vorkonditionierung')
    plt.plot(n, k_values_cond_max, label='mit Vorkonditionierung')
    plt.title('Anzahl Iterationen pro Dimension')
    plt.legend()
    plt.show()

    for i in range(0, 5):
        title = "Absolutes Residuum und Änderungsrate bzgl. r0 für N=" + str(n[i])
        plt.title(title)
        plt.xlabel("k-te Iteration")
        plt.ylabel("Residuum")
        plt.plot(k_values[i][:], r_values[i][:], '--',  label="abs. r ohne Vor")
        plt.plot(k_values[i][:], r_values_rate[i][:],  label="Änderungsrate ohne Vor")
        plt.plot(k_values_cond[i][:], r_values_cond[i][:], "--", label="abs. r mit Vor")
        plt.plot(k_values_cond[i][:], r_values_cond_rate[i][:], label="Änderungsrate mit Vor")
        plt.legend(loc="upper right")
        plt.show()
