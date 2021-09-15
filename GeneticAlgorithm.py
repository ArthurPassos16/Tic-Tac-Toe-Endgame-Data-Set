from scipy.special import expit
from sklearn.utils.estimator_checks import check_estimator
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets, unique_labels


class GeneticAlgorithm(BaseEstimator, ClassifierMixin):
    def __init__(self, pop_size: int = 100, p_crossover: float = 0.75, p_mutation: float = 0.5, bounds: tuple = (-10, 10),
                 mutate_bound: float = 0.01, elitism: float = 0.2, n_iter: int = 100, verbose: bool = False,
                 print_every: int = 10):
        self.pop_size = pop_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.bounds = bounds
        self.mutate_bound = mutate_bound
        self.elitism = elitism
        self.n_iter = n_iter
        self.verbose = verbose
        self.print_every = print_every

        pass

    def sigmoid(self, x: np.ndarray) -> np.float64:
        x = np.array(np.clip(x, -700, 700), dtype=np.float64)

        return 1/(1+np.exp(-x))

    def fitness_function(self, x: np.ndarray, W: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert(x.shape[1] == W.shape[0])
        assert(x.shape[0] == y.shape[0])

        y_hat = self.sigmoid(x.dot(W))
        vfunc = np.vectorize(lambda x: 1 if x > 0.5 else 0)
        y_hat = vfunc(y_hat)

        a = np.sum(np.apply_along_axis(
            lambda x: x == y, axis=0, arr=y_hat), axis=0)

        return a

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        generation = np.random.uniform(
            self.bounds[0], self.bounds[1], (X.shape[1], self.pop_size))

        self.classes_ = unique_labels(y)

        for j in range(self.n_iter):
            fitness = self.fitness_function(X, generation, y)
            if (self.verbose):
                if ((j % self.print_every) == 0):
                    print(f'Geração {j}')
                    print(f'Média   {np.mean(fitness)}')
                    print(f'Melhor  {np.max(fitness)}')

            prob_distribution = fitness/np.sum(fitness)
            ind = np.array(range(0, self.pop_size))

            # print(prob_distribution.shape)
            # print(ind.shape)
            temp = np.column_stack((prob_distribution, ind))
            temp = temp[np.argsort(temp[:, 0]), ]
            sorted_prob_distribution = temp[:, 0]

            best_index = self.pop_size - \
                np.maximum(1, int(self.elitism*self.pop_size))
            best_ind = range(best_index, self.pop_size)
            best_ind = temp[best_ind, 1]

            selected1 = np.random.choice(np.array(
                range(len(sorted_prob_distribution))), p=sorted_prob_distribution, size=self.pop_size)
            selected2 = np.random.choice(
                selected1, self.pop_size, replace=False)

            ind = np.asarray(temp[selected1, 1], dtype=int).tolist()
            selected = generation[:, ind]
            f1 = fitness[ind]
            ind = np.asarray(temp[selected2, 1], dtype=int).tolist()
            mates = generation[:, ind]
            f2 = fitness[ind]
            offspring = []
            r = np.random.uniform(0, 1, self.pop_size)
            for i in range(0, self.pop_size):
                if r[i] < self.p_crossover:
                    gamma = np.random.uniform(0, 1, selected.shape[0])
                    child = (1-gamma)*selected[:, i] + gamma*(mates[:, i])
                else:
                    if f1[i] < f2[i]:
                        child = mates[:, i]
                    else:
                        child = selected[:, i]
                offspring.append(child)

            r = np.random.uniform(0, 1, self.pop_size)

            mutations = np.random.uniform(-self.mutate_bound, self.mutate_bound,
                                          (selected.shape[0], selected.shape[1]))
            r = np.random.uniform(0, 1, (selected.shape[0], selected.shape[1]))
            r = r < self.p_mutation
            mutations = np.multiply(mutations, r)

            mutations_swapped = np.swapaxes(mutations, 0, 1)

            offspring = np.array(np.array(offspring) + mutations_swapped)
            offspring = offspring.reshape(
                offspring.shape[1], offspring.shape[0])

            new_fit = self.fitness_function(X, offspring, y)
            ind = range(0, self.pop_size)
            temp = np.column_stack((new_fit, ind))
            temp = temp[np.argsort(-temp[:, 0]), ]

            ind_replace = np.array(
                temp[best_index:self.pop_size, 1], dtype=int).tolist()

            next_gen = np.copy(offspring)

            next_gen[:, ind_replace] = generation[:, np.asarray(
                best_ind, dtype=int).tolist()]

            generation = np.copy(next_gen)

        fitness = self.fitness_function(X, generation, y)

        best_fit = generation[:, np.argmax(fitness)]

        self.best_fit_ = best_fit

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        y_hat = self.sigmoid(X.dot(self.best_fit_))
        vfunc = np.vectorize(lambda x: 1 if x > 0.5 else 0)
        y_hat = vfunc(y_hat)

        return y_hat

    def get_params(self, deep=False):
        return super().get_params(deep=deep)

    def set_params(self, **params):
        return super().set_params(**params)
