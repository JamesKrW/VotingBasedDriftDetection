import matplotlib.pyplot as plt

def plot(stream_window,change,path):
    y = stream_window
    x = [i for i in range(len(y))]

    plt.figure(figsize=(30, 6))

    # Plot the data
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Stream Data Plot')

    # Add vertical lines
    index = [i for i in change]
    for i in index:
        plt.axvline(x=i, color='red', linestyle='--')

    # Show the plot
    plt.savefig(f'{path}') 
    plt.show()