# main.py (gui.py)
import tkinter as tk
from tkinter import ttk
import threading, time, math
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import networkx as nx
from network_backend import Network
from chatgpt_interface import Poster
from scipy.stats import beta

# ---------------- Global Parameters ----------------
n_agents = 20
n_opinions = 2
init_X = np.column_stack((np.random.random(n_agents), beta.rvs(a=13, b=4, size=20)))
theta = 8
min_prob = 0.01
alpha_filter = 1
user_agents = [[0.5, 0.75]]
user_alpha = 0.5
strategic_agents = [[0, 1], [1, 0.5]]  # Two strategic agents.
strategic_theta = 2
clock_cycle = 3  # seconds per cycle
with open("key_file.txt", "r") as file:
   api_key = file.read()

opinion_axes = [
    {
        'name': 'Pineapple on Pizza',
        'pro': 'Pineapple would improve literally any pizza. You cannot go wrong with pineapple on pizza.',
        'con': 'Pineapple on pizza is disgusting. It is a disgrace that no one should have to eat or look at.'
    },
    {
        'name': 'Cats',
        'pro': 'Cats are the best friends of humans. They are the perfect pet with no flaws.',
        'con': 'Cats are mean, dirty, and undesirable in every possible way. Not even in the same conversation as dogs.'
    }
]

bot_names = np.array([
    "User", "Mason", "Luna", "Felix", "Nova", "Kai", "Sage", "Atlas", "Iris", "Finn",
    "Aria", "Nash", "Eden", "Leo", "Jade", "River", "Cora", "Axel", "Ivy", "Quinn"
])
# For this configuration with 20 agents and 2 strategic agents,
# the strategic agents are at indices: 18 and 19.

# ---------------- Helper Function ----------------
def compute_color(opinion):
    # Compute a color (hex) using the same formula as the visualization.
    x = opinion[0]
    y = opinion[1]
    intensity = math.sqrt(x*x + y*y) / math.sqrt(2)
    r = intensity * x + (1 - intensity)
    g = (1 - intensity)
    b = intensity * y + (1 - intensity)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

# ---------------- Scrollable Frame for Feed ----------------
class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, background="#ffffff")
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, background="#ffffff")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

# ---------------- GUI Class ----------------
class ChatGUI:
    def __init__(self, root):
        self.root = root
        # Set a fixed geometry so that the overall window is 1300x800.
        # New widths: Feed: 420px, Updates: 200px, Visualization: 680px.
        root.geometry("1300x800")
        self.root.title("Social Network Simulation")

        # Create three main frames with fixed widths.
        self.frame_feed = tk.Frame(root, bd=2, relief=tk.SUNKEN, width=420, height=800)
        self.frame_updates = tk.Frame(root, bd=2, relief=tk.SUNKEN, width=200, height=800)
        self.frame_vis = tk.Frame(root, bd=2, relief=tk.SUNKEN, width=680, height=800)

        self.frame_feed.grid(row=0, column=0, sticky="nsew")
        self.frame_updates.grid(row=0, column=1, sticky="nsew")
        self.frame_vis.grid(row=0, column=2, sticky="nsew")
        # Set grid column configurations.
        root.grid_columnconfigure(0, minsize=420, weight=1)
        root.grid_columnconfigure(1, minsize=200, weight=0)  # Fixed narrow updates column.
        root.grid_columnconfigure(2, minsize=680, weight=1)
        root.grid_rowconfigure(0, weight=1)

        # Prevent frames from resizing to fit content.
        self.frame_feed.grid_propagate(False)
        self.frame_updates.grid_propagate(False)
        self.frame_vis.grid_propagate(False)

        # ------------- Feed Section (Scrollable Posts) -------------
        self.feed_frame = ScrollableFrame(self.frame_feed)
        self.feed_frame.pack(fill=tk.BOTH, expand=True)
        # Bottom panel for user entry.
        self.entry_message = tk.Entry(self.frame_feed)
        self.entry_message.pack(side="left", fill="x", expand=True)
        self.button_send = tk.Button(self.frame_feed, text="Send", command=self.send_message)
        self.button_send.pack(side="right")

        # ------------- Updates Section -------------
        self.updates_feed = tk.Text(self.frame_updates, wrap="word", state="disabled", width=20)
        self.updates_feed.pack(fill=tk.BOTH, expand=True)

        # ------------- Visualization Section -------------
        self.notebook = ttk.Notebook(self.frame_vis)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.tab_opinions = tk.Frame(self.notebook)
        self.tab_connections = tk.Frame(self.notebook)
        self.notebook.add(self.tab_opinions, text="Opinions")
        self.notebook.add(self.tab_connections, text="Connections")

        self.fig_opinions = Figure(figsize=(4, 3), dpi=100)
        self.ax_opinions = self.fig_opinions.add_subplot(111)
        self.canvas_opinions = FigureCanvasTkAgg(self.fig_opinions, master=self.tab_opinions)
        self.canvas_opinions.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_connections = Figure(figsize=(4, 3), dpi=100)
        self.ax_connections = self.fig_connections.add_subplot(111)
        self.canvas_connections = FigureCanvasTkAgg(self.fig_connections, master=self.tab_connections)
        self.canvas_connections.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ------------- Initialize Network and Poster -------------
        self.network = Network(n_agents=n_agents, n_opinions=n_opinions, X=init_X.copy(), theta=theta, min_prob=min_prob,
                               alpha_filter=alpha_filter, user_agents=user_agents, user_alpha=user_alpha,
                               strategic_agents=strategic_agents, strategic_theta=strategic_theta)
        self.poster = Poster(api_key, opinion_axes)

        # For thread-safe handling of user posts.
        self.pending_user_post = None
        self.user_post_flag = False
        self.user_post_lock = threading.Lock()

        # Start the simulation loop in a background thread.
        self.running = True
        self.sim_thread = threading.Thread(target=self.simulation_loop, daemon=True)
        self.sim_thread.start()

    def send_message(self):
        message = self.entry_message.get().strip()
        if message:
            # For user posts, assume sender index 0.
            X, _, _ = self.network.get_state()
            opinion = X[0] if X.shape[0] > 0 else [0.5, 0.5]
            self.add_feed_message("User: " + message, sender_index=0, opinion_vector=opinion)
            with self.user_post_lock:
                self.pending_user_post = message
                self.user_post_flag = True
            self.entry_message.delete(0, tk.END)

    def add_feed_message(self, msg, sender_index, opinion_vector):
        # Determine border color:
        # User (index 0) always gold; strategic agents (indices >= n_agents - len(strategic_agents)) always black;
        # otherwise computed from opinion.
        if sender_index == 0:
            border_color = "gold"
        elif sender_index >= n_agents - len(strategic_agents):
            border_color = "black"
        else:
            border_color = compute_color(opinion_vector)

        # Create a frame for the post with a colored border.
        # Increase the wraplength to 400 so the post fills the feed box.
        post_frame = tk.Frame(self.feed_frame.scrollable_frame, bd=2, relief="solid",
                              highlightthickness=2, highlightbackground=border_color, padx=5, pady=5)
        label = tk.Label(post_frame, text=msg, wraplength=400, justify="left")
        label.pack(fill="x")
        post_frame.pack(fill="x", padx=5, pady=5)
        # Auto-scroll to the bottom.
        self.feed_frame.canvas.update_idletasks()
        self.feed_frame.canvas.yview_moveto(1.0)

    def update_updates(self, updates_list):
        def update():
            self.updates_feed.configure(state="normal")
            self.updates_feed.delete("1.0", tk.END)
            for u in updates_list:
                self.updates_feed.insert(tk.END, u + "\n")
            self.updates_feed.configure(state="disabled")
        self.root.after(0, update)

    def update_visualizations(self, X, A):
        def update_figures():
            # --- Opinions Tab ---
            self.ax_opinions.clear()
            x = X[:, 0]
            y = X[:, 1]
            colors = []
            for i in range(len(x)):
                if i == 0:
                    colors.append("gold")
                elif i >= n_agents - len(strategic_agents):
                    colors.append("black")
                else:
                    colors.append(compute_color([x[i], y[i]]))
            self.ax_opinions.scatter(x, y, c=colors, edgecolors="none", s=50)
            # Draw circles around highlighted nodes.
            if len(x) > 0:
                self.ax_opinions.scatter(x[0], y[0], facecolors="none", edgecolors="gold", s=150, linewidths=2)
            for idx in range(n_agents - len(strategic_agents), n_agents):
                if idx < len(x):
                    self.ax_opinions.scatter(x[idx], y[idx], facecolors="none", edgecolors="black", s=150, linewidths=2)
            self.ax_opinions.set_title("Distribution of Opinions")
            self.ax_opinions.set_xlabel("Opinion on Pineapple on Pizza")
            self.ax_opinions.set_ylabel("Opinion on Cats")
            self.ax_opinions.set_xticks(np.arange(0, 1.05, 0.1))
            self.ax_opinions.set_yticks(np.arange(0, 1.05, 0.1))
            self.canvas_opinions.draw()

            # --- Connections Tab ---
            self.ax_connections.clear()
            G = nx.from_numpy_array(np.array(A))
            colors_conn = []
            for i in range(len(X)):
                if i == 0:
                    colors_conn.append("gold")
                elif i >= n_agents - len(strategic_agents):
                    colors_conn.append("black")
                else:
                    colors_conn.append(compute_color(X[i]))
            pos = nx.spring_layout(G, seed=1)
            nx.draw(G, pos=pos, node_color=colors_conn, node_size=50, width=0.25,
                    edgecolors='none', ax=self.ax_connections)
            highlight = [0] + [i for i in range(n_agents - len(strategic_agents), n_agents)]
            if highlight:
                nx.draw_networkx_nodes(G, pos, nodelist=highlight,
                                       node_color=[colors_conn[i] for i in highlight],
                                       edgecolors=["gold" if i==0 else "black" for i in highlight],
                                       node_size=100, linewidths=2.5, ax=self.ax_connections)
            self.ax_connections.set_title("Network Connections")
            self.ax_connections.axis('off')
            self.canvas_connections.draw()
        self.root.after(0, update_figures)

    def simulation_loop(self):
        global clock_cycle, bot_names, strategic_agents
        while self.running:
            start_time = time.time()
            # Flush the ChatGPT history at the beginning of each cycle.
            self.poster.flush_history()
            # Update the network state.
            X, A, timestep = self.network.update_network()
            # Update visualizations.
            self.update_visualizations(X, A)
            # Check for a pending user post.
            with self.user_post_lock:
                if self.user_post_flag:
                    try:
                        opinion_vector = self.poster.analyze_post(self.pending_user_post)
                    except Exception as e:
                        opinion_vector = [0.5, 0.5]  # fallback neutral
                    self.network.add_user_opinion(opinion_vector, user_index=0)
                    # Create updates based on connections.
                    updates = []
                    for i in range(A.shape[0]):
                        if i != 0:
                            if A[0, i] == 1:
                                updates.append(f"{bot_names[i]} read your post.")
                            else:
                                updates.append(f"{bot_names[i]} ignored your post.")
                    self.update_updates(updates)
                    self.user_post_flag = False
                    self.pending_user_post = None
            # Process friend posts – each friend posts only once per cycle.
            friend_indices = [i for i in range(A.shape[0]) if i != 0 and A[0, i] == 1]
            num_friends = len(friend_indices)
            delay = clock_cycle / num_friends if num_friends > 0 else clock_cycle
            for friend in friend_indices:
                time.sleep(delay)
                friend_opinion = X[friend]
                friend_name = bot_names[friend]
                try:
                    post = self.poster.generate_post(friend_name, friend_opinion)
                except Exception as e:
                    post = "Default post."
                self.add_feed_message(f"{friend_name}: {post}", sender_index=friend, opinion_vector=friend_opinion)
            elapsed = time.time() - start_time
            if elapsed < clock_cycle:
                time.sleep(clock_cycle - elapsed)

def main():
    root = tk.Tk()
    gui = ChatGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

