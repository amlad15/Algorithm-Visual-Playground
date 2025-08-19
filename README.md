Algorithm Visual Playground 🎨✨

Hi there! 👋

I’ve been working on this little project where I wanted to see algorithms in action instead of just reading code. So, I built this Streamlit app that lets you play around with sorting, searching, and pathfinding algorithms — and actually watch them unfold step by step. 🚀

💡 What I built

I have a sorting visualizer where you can pick algorithms like Bubble Sort, Merge Sort, Quick Sort, etc. and watch the bars shuffle around until everything lines up.

I also added a pathfinding section — things like BFS, DFS, A*, and Dijkstra’s. I found it super fun to drop walls on a grid and see the algorithm figure out how to get from start to finish.

On top of that, there’s a graph visualizer where you can run search algorithms and see nodes light up as the algorithm explores.

Basically, it’s like a playground where algorithms come alive.

🔧 How I made it

I used Python + Streamlit for the app framework.

Matplotlib and NumPy handle the visuals for grids and graphs.

At first, I ran into some bugs with function calls (like render_grid needing specific args instead of just a state object). I found that making the function “state-friendly” simplified things a lot.

I also learned that Streamlit’s session_state is super handy for remembering where the user is in the simulation.

🚀 Running it locally

If you want to try it out:

Clone this repo:

git clone https://github.com/yourusername/algorithm-visual-playground.git
cd algorithm-visual-playground


Install the requirements:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


That should open it up in your browser at http://localhost:8501
.

📸 What it looks like

<img width="1906" height="820" alt="{0160DB1B-24EB-4710-8055-0F24C66623BE}" src="https://github.com/user-attachments/assets/84c16c27-5285-40e9-af7b-c53125491378" />
<img width="1882" height="843" alt="{C80D4EBD-4CEC-4B94-8FE1-58EB8CD691F9}" src="https://github.com/user-attachments/assets/8c099a3f-9e43-4cb4-8830-d05e16ced981" />



🤔 What I learned

I realized algorithms are way easier to understand once you can see them.

Debugging visualization code is its own puzzle — like at one point I had a TypeError because I forgot to unpack grid, start, end from state.

Streamlit makes it feel like you’re building an app really fast, but it’s still just Python under the hood, so you get the best of both worlds.

🛠️ Future ideas

Add maze generation so you can let the algorithm solve a random maze.

Add more sorting algorithms (like Heap Sort).

Maybe even a step-by-step playback button instead of just autoplay.

🙌 Wrapping up

This was one of those projects where I started with “lemme just try sorting” and ended up going down the rabbit hole of pathfinding and grids. I found that once you get a little visualization running, it’s addictive to keep adding new algorithms.

If you want to play with algorithms, or just like watching them dance, give it a try and let me know what you think!
