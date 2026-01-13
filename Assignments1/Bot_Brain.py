import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
import heapq
import math
import sqlite3
from PIL import Image, ImageTk

def init_db():
    conn = sqlite3.connect('campus.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS buildings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS paths (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        destination TEXT NOT NULL,
        distance REAL NOT NULL,
        FOREIGN KEY(source) REFERENCES buildings(name),
        FOREIGN KEY(destination) REFERENCES buildings(name)
    )
    ''')
    cursor.execute('SELECT COUNT(*) FROM buildings')
    if cursor.fetchone()[0] == 0:
        buildings_data = [
            ('Main Gate', 'Main entrance to the campus, security checkpoints.'),
            ('Enquiry Gate', 'Information desk open from 8 AM to 5 PM.'),
            ('Block A', 'Includes Registrar Office, Library, and Canteen.'),
            ('Auditorium', 'Events and seminar hall.'),
            ('Register Office', 'Student registration and certificates.'),
            ('Library', 'Open 8 AM to 8 PM, thousands of books and digital resources.'),
            ('Canteen', 'Breakfast, lunch, and snacks available till 9 PM.'),
            ('Block B', 'Engineering classrooms and faculty offices.'),
            ('Engineering Building', 'Departments of Computer Science and Electronics.'),
            ('Food Court', 'Multiple food vendors and laundry services.'),
            ('Hostel', 'Student accommodation with mess and common rooms.'),
            ('Sports Area', 'Volleyball and basketball courts.'),
            ('Cricket Ground', 'Outdoor cricket ground for practice and matches.'),
        ]
        paths_data = [
            ('Main Gate', 'Enquiry Gate', 50),
            ('Enquiry Gate', 'Block A', 25),
            ('Block A', 'Auditorium', 10),
            ('Block A', 'Register Office', 15),
            ('Block A', 'Library', 20),
            ('Block A', 'Canteen', 25),
            ('Block A', 'Block B', 80),
            ('Block B', 'Food Court', 80),
            ('Block B', 'Engineering Building', 10),
            ('Food Court', 'Hostel', 100),
            ('Food Court', 'Sports Area', 120),
            ('Hostel', 'Sports Area', 600),
            ('Sports Area', 'Cricket Ground', 80),
        ]
        cursor.executemany('INSERT INTO buildings (name, description) VALUES (?, ?)', buildings_data)
        cursor.executemany('INSERT INTO paths (source, destination, distance) VALUES (?, ?, ?)', paths_data)
        conn.commit()
    return conn

def fetch_graph_from_db(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT name FROM buildings')
    buildings = [row[0] for row in cursor.fetchall()]
    graph = {b: [] for b in buildings}
    cursor.execute('SELECT source, destination, distance FROM paths')
    for source, dest, dist in cursor.fetchall():
        graph[source].append((dest, dist))
        graph[dest].append((source, dist))
    return graph

def fetch_building_info(conn, building_name):
    cursor = conn.cursor()
    cursor.execute('SELECT description FROM buildings WHERE name=?', (building_name,))
    res = cursor.fetchone()
    return res[0] if res else "No info available for this building."

def bfs(graph, start, goal):
    queue = deque([(start, [start])])
    explored = set()
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        explored.add(node)
        for nbr, _ in graph[node]:
            if nbr not in explored and nbr not in [p[0] for p in queue]:
                queue.append((nbr, path + [nbr]))
    return None

def calc_distance(graph, path):
    dist = 0
    for i in range(len(path) - 1):
        for nbr, w in graph[path[i]]:
            if nbr == path[i+1]:
                dist += w
                break
    return dist

def simple_chatbot(user_input, campus_graph, conn):
    user_input = user_input.lower()
    if any(greet in user_input for greet in ['hello', 'hi', 'hey']):
        return "Hello! How can I assist you with campus navigation today?"

    if "info" in user_input or "about" in user_input:
        for building in campus_graph.keys():
            if building.lower() in user_input:
                return fetch_building_info(conn, building)
        return "Which building would you like information about?"

    if "navigate" in user_input or "how to reach" in user_input or "route" in user_input:
        return "Please enter your start and destination locations in the UI and press 'Find Path' for navigation."

    return "Sorry, I didn't understand. You can ask about building info or navigation."

class CampusNavigatorUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bot Brain: Your Personal Campus Guide with Chatbot")
        self.root.geometry("950x650")  # increased height for chat area
        self.conn = init_db()
        self.campus_graph = fetch_graph_from_db(self.conn)
        self.buildings = list(self.campus_graph.keys())
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Bot Brain: Your Personal Campus Guide",
                 font=("Arial", 18, "bold")).pack(pady=10)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        tk.Label(left_frame, text="Start:").grid(row=0, column=0, padx=5, sticky="e")
        self.start_var = tk.StringVar(value=self.buildings[0])
        ttk.Combobox(left_frame, textvariable=self.start_var,
                     values=self.buildings, state="readonly").grid(row=0, column=1, padx=5)

        tk.Label(left_frame, text="Destination:").grid(row=1, column=0, padx=5, sticky="e")
        self.dest_var = tk.StringVar(value=self.buildings[1])
        ttk.Combobox(left_frame, textvariable=self.dest_var,
                     values=self.buildings, state="readonly").grid(row=1, column=1, padx=5)

        tk.Button(left_frame, text="Find Path", command=self.find_path,
                  bg="#007ACC", fg="white", font=("Arial", 12)).grid(row=2, column=0, columnspan=2, pady=15)

        tk.Label(left_frame, text="Result:").grid(row=3, column=0, sticky="nw", pady=(20, 0))
        self.result_text = tk.Text(left_frame, height=10, width=40)
        self.result_text.grid(row=4, column=0, columnspan=2, pady=5)

        # Chatbot UI components
        tk.Label(left_frame, text="Chatbot:").grid(row=5, column=0, sticky="w", pady=(20,0))
        self.chat_entry = tk.Entry(left_frame, width=40)
        self.chat_entry.grid(row=6, column=0, columnspan=2, pady=5)
        self.chat_send_btn = tk.Button(left_frame, text="Send", command=self.chatbot_response)
        self.chat_send_btn.grid(row=7, column=0, columnspan=2, pady=10)
        self.chat_text = tk.Text(left_frame, height=8, width=40, state=tk.DISABLED)
        self.chat_text.grid(row=8, column=0, columnspan=2, pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, padx=15)

        try:
            img = Image.open("CampusGraph_Edited.jpg")
            img = img.resize((420, 420), Image.Resampling.LANCZOS)
            self.map_img = ImageTk.PhotoImage(img)
            tk.Label(right_frame, image=self.map_img).pack()
            tk.Label(right_frame, text="Campus Map", font=("Arial", 12)).pack()
        except Exception as e:
            tk.Label(right_frame, text="Campus map image not found.").pack()
            print("Error loading map image:", e)

    def chatbot_response(self):
        user_msg = self.chat_entry.get()
        if not user_msg.strip():
            return
        self.chat_entry.delete(0, tk.END)
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, "You: " + user_msg + "\n")
        bot_msg = simple_chatbot(user_msg, self.campus_graph, self.conn)
        self.chat_text.insert(tk.END, "Bot: " + bot_msg + "\n\n")
        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def find_path(self):
        start = self.start_var.get()
        dest = self.dest_var.get()

        if start == dest:
            messagebox.showwarning("Invalid Input", "Start and destination must differ.")
            return

        path = bfs(self.campus_graph, start, dest)
        dist = calc_distance(self.campus_graph, path) if path else None

        self.result_text.delete(1.0, tk.END)

        if path:
            time_est = round(dist / 1.2, 1)  # Estimate walking speed ~1.2 m/s
            building_info = fetch_building_info(self.conn, dest)
            info = f"Path: {' → '.join(path)}\n"
            info += f"Distance: {dist} m\n"
            info += f"Est. Walking Time: {time_est} seconds\n\n"
            info += f"Destination Info:\n{building_info}"
            self.result_text.insert(tk.END, info)
        else:
            self.result_text.insert(tk.END, "No path found between selected locations.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    CampusNavigatorUI().run()





""" Queries You Can Ask the Chatbot:
"Hi"

"Hello"

"Tell me about Main Gate"

"What is the Library?"

"Give me info about Hostel"

"How do I navigate to Cricket Ground?"

"Can you tell me the route to Auditorium?"
"""
