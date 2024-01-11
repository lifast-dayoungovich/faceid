import matplotlib.pyplot as plt
import numpy as np

# Assuming 'history' is a dictionary containing 'loss' and 'accuracy' keys
history_25_epochs = {
    'loss': [0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03,
             0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
             0.01, 0.01, 0.01, 0.01, 0.01],
    'accuracy': [0.8, 0.85, 0.9, 0.92, 0.95, 0.96, 0.97, 0.98, 0.98, 0.99,
                 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99,
                 0.99, 0.99, 0.99, 0.99, 1.0]
}

def plot_training_history(history_a,history_b):
    epochs = len(history_a['loss'])
    plt.figure(figsize=(12, 4))

    x = range(1, epochs + 1)
    y_loss_a = history_a['loss']
    y_accuracy_a = history_a['accuracy']
    y_loss_b = history_b['loss']
    y_accuracy_b = history_b['accuracy']

    # Plotting loss
    ax = plt.subplot(1, 2, 1)
    ax.set_xlim([min(x), max(x)+1])
    ax.plot(x, y_loss_a, marker='o')
    ax.plot(x, y_loss_b, marker='o', color='r')
    # ax.hlines(y=y_loss[-1], xmin=-2, xmax=x[-1], color='r', linestyles=(0.2,[0.2,0.3]))
    # plt.vlines(x=mu, ymin=min(y), ymax=max(y), color='r', marker='-')
    ax.set_title('Помилка Тренуваня')
    ax.set_xlabel('Епохи')
    ax.set_ylabel('Помилка')
    ax.legend(['SGD', 'Adam'])

    # Plotting accuracy
    ax = plt.subplot(1, 2, 2)
    ax.set_xlim([min(x), max(x)+1])
    ax.plot(x, y_accuracy_a, marker='o')
    ax.plot(x, y_accuracy_b, marker='o', color='r')
    # ax.hlines(y=y_accuracy[-1], xmin=0, xmax=x[-1], color='r', linestyles='dotted')
    ax.set_title('Точність(Accuracy) Тренування')
    ax.set_xlabel('Епохи')
    ax.set_ylabel('Точність')
    ax.legend(['SGD', 'Adam'])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Plot for 25 epochs
    # plot_training_history(history_25_epochs, epochs=25)

    def generate_random_history(epochs, bias=0.86):
        # Generating random values for loss and accuracy
        x = np.arange(1, epochs + 1)
        loss_values = (36 + 7 * np.random.random(epochs)) * np.ones(epochs)/x / (bias*100-85)
        accuracy_values = -np.ones(epochs)/x + np.random.random(epochs)*0.05 + bias

        # Creating a dictionary with generated values
        history = {'loss': loss_values, 'accuracy': accuracy_values}

        return history


    # Example usage for 25 epochs
    epochs_25 = 50
    history_25_epochs_a = generate_random_history(epochs_25)
    history_25_epochs_b = generate_random_history(epochs_25, bias=0.88)
    plot_training_history(history_25_epochs_a, history_25_epochs_b)
