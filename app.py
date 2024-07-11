from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
CORS(app)

# Load the model and data
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

facts = [
    "Hardik Jaiswal is a python developer with a keen interest in AI, ML and robotics.",
"Hardik Jaiswal is a python developer with a keen interest in AI, ML and robotics.",
"Hardik Jaiswal is a python developer with a keen interest in AI, ML and robotics.",
"Hardik's full name is Hardik Jaiswal.",
"Hardik's full name is Hardik Jaiswal.",
"Hardik Jaiswal is a python developer with a keen interest in AI, ML and robotics.",
"Hardik Jaiswal is a python developer with a keen interest in AI, ML and robotics.",
"Hardik is based in Kolkata, India.",
"Hardik lives in Kolkata, India.",
"Hardik graduated from high school in the year 2024 and is currently a first year student at Newton School Of Technology.",
"Hardik graduated from high school in the year 2024 and is currently pursuing his Bachelor's at Newton School Of Technology.",
"Hardik graduated from high school in the year 2024.",
"Hardik graduated from high school in the year 2024.",
"Hardik is pursuing his Bachelor's at Newton School Of Technology.",
"Hardik is pursuing his Bachelor's at Newton School Of Technology.",
"Hardik is interested in AI, ML and Robotics. You can know more about him in the about page.",
"Hardik has skills in software development and web development. Head over to the about page for more information.",
"Hardik has worked on a project named Peripheral Killing System which eliminates the use of mouse and keyboard, and replaces them with human gestures.",
"Hardik has worked on a project named Peripheral Killing System which eliminates the use of mouse and keyboard, and replaces them with human gestures.",
"Hardik has worked on projects like Peripheral Killing System, Melody.CLI, Hardik.AI and Devcraft. Know more in the projects page.",
"Hardik has worked on projects like Peripheral Killing System, Melody.CLI, Hardik.AI and Devcraft. Know more in the projects page.",
"Hardik is proficient in Python, HTML, CSS, JavaScript, Java, ReactJS, React Native and is currently learning AI and DSA.",
"Hardik is proficient in Linux, Git, Visual Studio Code, and opensource.",
"Hardik uses technologies such as Python and JavaScript.",
"Hardik's career goal is to become a skilled AI and ML specialist while also maintaining strong web development skills.",
"Hardik's personal motto is 'Continuous learning and improvement are key to success.'",
"Hardik's hobbies include playing cricket, coding, and playing video games.",
"Hardik enjoys reading books on technology, science fiction, and personal development.",
"Hardik enjoys watching sci-fi movies, action films, and documentaries.",
"Hardik's favorite programming language is Python because of its simplicity and versatility.",
"Hardik has completed courses in Python programming, web development, and AI-ML.",
"Hardik has a strong interest in AI and ML and has completed several online courses in these fields. He is currently working on integrating AI features into his portfolio website.",
"Hardik has worked on several web development projects, including a personal portfolio website, a blog platform, and a small e-commerce site.",
"Hardik is currently learning ReactJS and has built a few components for his portfolio website. He is also learning to use ReactJS with NextJS for server-side rendering.",
"Hardik has used TailwindCSS to style his portfolio website and appreciates its utility-first approach for building custom designs quickly.",
"Hardik enjoys working on projects that challenge him to learn new technologies and solve complex problems, especially in the fields of AI, ML, and web development.",
"Hardik's favorite tools for development include VS Code for coding, Git for version control, and Docker for containerization.",
"Hardik has contributed to a few open-source projects on GitHub, primarily focused on Python libraries and web development tools.",
"Hardik's favorite project is his personal portfolio website because it allowed him to showcase his skills and creativity while learning new technologies.",
"Hardik's strengths include problem-solving, quick learning, and a strong foundation in programming and web development.",
"One of Hardik's weaknesses is that he can sometimes get too focused on small details, but he is working on improving his time management and prioritization skills.",
"Hardik is motivated by the desire to learn new things and solve challenging problems. He also enjoys seeing the impact of his work on others.",
"Hardik handles stress and pressure by staying organized, taking breaks when needed, and maintaining a healthy work-life balance.",
"Hardik's approach to learning new technologies involves hands-on practice, building small projects, and seeking out online resources and courses.",
"Hardik has experience using Git for version control, including branching, merging, and collaborating on projects with others using GitHub.",
"Hardik has used Docker to containerize applications and manage development environments, which helps ensure consistency across different systems.",
"Hardik's future plans include continuing to develop his skills in AI and ML, working on more complex projects, and eventually pursuing a career in the tech industry.",
"What Hardik likes most about programming is the ability to create something from nothing and solve real-world problems through code.",
"What Hardik likes most about web development is the opportunity to create interactive and visually appealing websites that provide great user experiences.",
"Hardik's favorite online resources for learning include freeCodeCamp, Coursera, and YouTube channels like Traversy Media and Academind.",
"Hardik speaks English and Hindi fluently.",
"Hardik is proud of completing several online courses in AI and ML, building his personal portfolio website, and contributing to open-source projects.",
"Hardik stays updated with the latest trends in technology by following tech blogs, subscribing to newsletters, and participating in online communities and forums.",
"Hardik is part of online communities like GitHub, Stack Overflow, and various Discord servers related to programming and technology.",
"Hardik is inspired by innovative technologies and the potential they have to solve real-world problems and improve lives.",
"Hardik's favorite aspect of AI and ML is the ability to build systems that can learn from data and make intelligent decisions.",
"Hardik's goals for his portfolio website are to showcase his skills, share his projects, and provide an interactive platform for visitors to learn more about him.",
"Hardik approaches problem-solving by breaking down the problem into smaller, manageable parts, researching possible solutions, and testing different approaches until he finds the best one.",
"Hardik is currently learning NextJS to enhance his ReactJS skills and take advantage of its server-side rendering capabilities.",
"Hardik manages his time effectively by setting clear goals, prioritizing tasks, and using tools like calendars and to-do lists to stay organized.",
"From his previous projects, Hardik has learned the importance of planning, testing, and iterating to achieve the best results.",
"Hardik's advice to someone starting in programming is to practice consistently, build projects, and never hesitate to seek help from the community.",
"Challenges Hardik has faced in his projects include debugging complex issues, managing time effectively, and keeping up with rapidly evolving technologies.",
"Hardik stays motivated by setting achievable goals, celebrating small wins, and constantly seeking opportunities to learn and grow.",
"Hardik's approach to debugging code involves carefully reading error messages, methodically testing parts of the code, and using debugging tools to trace the source of issues.",
"Hardik's favorite software tool for coding is Visual Studio Code because of its versatility and extensive range of extensions.",
"For backend development, Hardik prefers using Django due to its robustness and the batteries-included philosophy.",
"Hardik ensures code quality by writing clean, maintainable code, performing regular code reviews, and using linters and static analysis tools.",
"Hardik has experience working with both relational databases like MySQL and PostgreSQL, and NoSQL databases like MongoDB.",
"Hardik approaches learning a new programming language by first understanding its syntax and core concepts, then building small projects to practice and deepen his understanding.",
"Hardik uses agile methodologies in his projects, focusing on iterative development, regular feedback, and continuous improvement.",
"Hardik's favorite aspect of web development is the ability to create dynamic and interactive applications that enhance user experiences.",
"Hardik's approach to designing user interfaces involves prioritizing user experience, using intuitive design patterns, and ensuring responsiveness across different devices.",
"Hardik has worked with ReactJS only.",
"To keep his skills up to date, Hardik regularly participates in online courses, attends webinars, and follows industry news.",
"Hardik collaborates with other developers using version control systems like Git, communication tools like Slack, and project management tools like Jira.",
"Hardik has used version control systems such as Git and SVN.",
"Hardik tests his code using unit tests, integration tests, and end-to-end tests to ensure it functions as expected and handles edge cases.",
"Hardik prefers a development environment that is customizable and efficient, typically using VS Code with various extensions tailored to his workflow.",
"Hardik has experience using cloud services like AWS and Google Cloud for deploying and managing applications.",
"Hardik follows web development methodologies such as responsive design, mobile-first development, and progressive enhancement.",
"Hardik's goals for the future include mastering advanced AI-ML techniques, contributing to significant open-source projects, and developing innovative tech solutions.",
"Hardik has worked on AI-ML projects such as image recognition systems, sentiment analysis tools, and predictive analytics models.",
"Hardik's approach to project management involves setting clear objectives, breaking down tasks into manageable parts, and regularly reviewing progress.",
"Hardik's favorite AI-ML library is TensorFlow due to its flexibility and extensive community support.",
"Hardik keeps his code secure by following best practices such as using strong passwords, encrypting sensitive data, and regularly updating dependencies.",
"Hardik has experience designing and consuming RESTful APIs, as well as working with GraphQL for more flexible data querying.",
"Hardik uses platforms like GitHub, Heroku, and Vercel for hosting his projects and ensuring they are accessible online.",
"Hardik ensures application performance by optimizing code, using efficient algorithms, and conducting thorough performance testing.",
"Hardik has experience with cloud-native development, including using Kubernetes for container orchestration and deploying microservices architectures.",
"Hardik handles version control conflicts by carefully reviewing changes, communicating with team members, and using merge tools to resolve issues.",
"Hardik's favorite productivity tool is Notion because of its versatility in managing notes, tasks, and projects.",
"Hardik approaches writing documentation by clearly explaining the purpose, functionality, and usage of code, and including examples and diagrams when necessary.",
"Hardik has experience setting up CI/CD pipelines using tools like Jenkins, GitHub Actions, and Travis CI to automate testing and deployment.",
"Hardik enjoys algorithmic coding challenges that require logical thinking and optimization, often participating in platforms like LeetCode and Codewars.",
"Hardik contributes to the tech community by writing blog posts, giving talks at meetups, and mentoring aspiring developers.",
"Hardik has completed online courses on platforms like Coursera, Udemy, and edX, covering topics in Python programming, AI, and web development.",
"Hardik prefers using Visual Studio Code for general development and PyCharm for Python-specific projects.",
"Hardik's approach to optimizing web performance includes minimizing HTTP requests, using lazy loading, and leveraging browser caching.",
"Hardik handles cross-browser compatibility issues by testing his websites on multiple browsers and using polyfills and fallbacks for unsupported features.",
"Hardik has some experience with mobile development using frameworks like React Native to build cross-platform applications.",
"Hardik is currently learning TypeScript to enhance his JavaScript development skills and Rust for systems programming.",
"Hardik's approach to learning a new framework involves reading the official documentation, building small projects, and exploring community resources.",
"Hardik follows coding practices such as writing clean and maintainable code, adhering to SOLID principles, and conducting regular code reviews.",
"Hardik has experience with serverless architecture, using services like AWS Lambda and Google Cloud Functions to build scalable applications.",
"Hardik ensures accessibility by following WCAG guidelines, using semantic HTML, and testing with screen readers and other assistive technologies.",
"Hardik's approach to testing web applications involves writing unit tests, integration tests, and using tools like Selenium for end-to-end testing.",
"Hardik has experience with data visualization using libraries like D3.js and Plotly to create interactive charts and graphs.",
"Hardik enjoys the collaborative aspect of open-source projects and the opportunity to contribute to tools that benefit the wider developer community.",
"Hardik approaches learning complex topics by breaking them down into smaller parts, studying relevant materials, and applying concepts through practical projects.",
"Hardik's favorite aspect of being a developer is the constant opportunity for learning and the ability to create impactful solutions through code.",
"Hardik handles feedback by listening carefully, considering the suggestions, and making improvements to enhance the quality of his work.",
"Hardik has experience with Agile development, working in sprints, participating in stand-up meetings, and using tools like Jira for project management.",
"Hardik's approach to continuous learning involves setting aside regular time for study, exploring new technologies, and participating in online courses and workshops.",
"Hardik's favorite development environment setup includes using a dual-monitor setup, a mechanical keyboard, and a comfortable chair for long coding sessions.",
"Hardik approaches refactoring code by first understanding the existing functionality, then incrementally making improvements to enhance readability and performance.",
"Hardik has experience with backend frameworks like Django, Flask, and Express.js for building robust and scalable server-side applications.",
"Hardik stays focused while working by using techniques like the Pomodoro Technique, minimizing distractions, and taking regular breaks to maintain productivity.",
"Hardik's approach to solving coding challenges involves understanding the problem statement, devising a plan, and writing efficient code to implement the solution.",
"Hardik uses tools like GitHub for version control, Slack for communication, and Trello for task management to collaborate effectively with his team.",
"Hardik's favorite programming language is Python because of its simplicity, readability, and extensive libraries for AI and ML development.",
"Hardik approaches debugging code by carefully reading error messages, using breakpoints, and systematically checking the logic to identify and fix issues.",
"Hardik has experience with front-end frameworks like ReactJS and Angular, using them to build dynamic and responsive user interfaces.",
"Hardik handles tight deadlines by prioritizing tasks, staying organized, and communicating effectively with team members to ensure timely completion.",
"Hardik ensures code quality by writing clean, maintainable code, conducting code reviews, and using automated testing tools to catch bugs early.",
"Hardik has experience with DevOps practices such as continuous integration, continuous deployment, and infrastructure as code using tools like Terraform.",
"Hardik handles learning new technologies by staying curious, exploring documentation, building small projects, and seeking help from the developer community.",
"Hardik's favorite software development methodology is Agile because it promotes flexibility, collaboration, and iterative progress.",
"Hardik has experience with database management using both SQL and NoSQL databases, including MySQL, PostgreSQL, and MongoDB.",
"Hardik approaches designing user interfaces by focusing on user experience, using wireframes and prototypes, and iterating based on user feedback.",
"Hardik has experience with machine learning frameworks such as TensorFlow, Keras, and PyTorch, using them to develop and train various models.",
"Hardik handles stress by taking regular breaks, practicing mindfulness, and maintaining a healthy work-life balance to stay focused and productive.",
"Hardik approaches code optimization by profiling the code to identify bottlenecks, optimizing algorithms, and reducing unnecessary computations.",
"Hardik uses tools like Prometheus and Grafana for monitoring applications, tracking performance metrics, and alerting on issues.",
"Hardik approaches collaboration on open-source projects by actively participating in discussions, contributing code, and helping with documentation and issue resolution.",
"Hardik has experience with API development, designing RESTful and GraphQL APIs, and ensuring they are secure and efficient.",
"Hardik handles changes in project requirements by staying flexible, communicating with stakeholders, and adapting his approach to meet the new objectives.",
"Hardik has experience with containerization using Docker, creating containerized applications, and orchestrating them with Kubernetes.",
"Hardik's approach to writing clean code involves following naming conventions, keeping functions and classes small, and writing meaningful comments.",
"Hardik stays updated with industry trends by reading tech blogs, following thought leaders on social media, and attending webinars and conferences.",
"Hardik has experience with microservices architecture, designing and deploying microservices, and ensuring they communicate effectively using APIs.",
"Hardik approaches troubleshooting issues in production by analyzing logs, using monitoring tools, and systematically isolating and resolving the root cause.",
"Hardik has experience with big data technologies like Hadoop and Spark, using them for data processing and analytics on large datasets.",
"Hardik approaches code reviews by providing constructive feedback, focusing on code quality, and ensuring that best practices are followed.",
"Hardik has experience with real-time data processing using tools like Apache Kafka and Flink to handle and analyze data streams.",
"Hardik handles working with legacy code by understanding the existing codebase, refactoring when necessary, and carefully adding new features.",
"Hardik has experience with machine learning model deployment, using tools like TensorFlow Serving and Flask to serve trained models in production.",
"Hardik approaches learning from failures by analyzing what went wrong, understanding the lessons learned, and applying those insights to future projects.",
"Hardik has experience with natural language processing (NLP), using libraries like NLTK and spaCy to build applications like chatbots and text classifiers.",
"Hardik approaches integrating third-party services by thoroughly reading the documentation, using SDKs and APIs, and ensuring proper error handling.",
"Hardik has experience with version control systems like Git, using them for code management, collaboration, and maintaining project history.",
"Hardik handles multitasking on different projects by prioritizing tasks, managing his time effectively, and focusing on one project at a time to ensure quality.",
"Hardik's approach to writing unit tests involves creating tests for individual components, ensuring coverage for various scenarios, and using testing frameworks like Jest and pytest.",
"Hardik ensures data privacy and security by implementing encryption, using secure authentication methods, and following best practices for data handling.",
"Hardik has studied AI ethics and fairness, understanding the importance of unbiased data, transparent algorithms, and the ethical implications of AI applications.",
"Hardik handles rapid changes in technology by staying adaptable, continuously learning, and experimenting with new tools and frameworks to stay current.",
"Hardik's approach to mentoring junior developers involves providing guidance, sharing knowledge, encouraging best practices, and fostering a supportive learning environment.",
"Hardik's favorite Python library is NumPy because of its powerful array operations and mathematical functions.",
"Hardik has experience with Flask and Django for web development in Python.",
"Hardik appreciates Flask's simplicity and flexibility, making it easy to build web applications quickly.",
"Hardik likes Django for its batteries-included approach, which provides built-in features like authentication, admin panel, and ORM.",
"Hardik handles errors in Python using try-except blocks and logging to identify and debug issues effectively.",
"Hardik secures Python applications by implementing best practices like input validation, using secure libraries, and regularly updating dependencies.",
"Hardik has worked with SQLite and PostgreSQL databases in Python projects.",
"Hardik deploys Python applications using Docker containers and cloud services like AWS Elastic Beanstalk for scalable deployments.",
"Hardik ensures Python code quality by writing unit tests, using pylint for code analysis, and following PEP 8 guidelines.",
"Hardik uses Jenkins and GitHub Actions for continuous integration and deployment of Python applications.",
"Hardik likes the MVC (Model-View-Controller) and Singleton design patterns for organizing and simplifying Python code.",
"Hardik uses Matplotlib and Seaborn for data visualization tasks in Python.",
"Hardik optimizes Python code by profiling performance with tools like cProfile, optimizing algorithms, and using caching techniques.",
"Hardik prefers using BeautifulSoup and Scrapy for web scraping tasks in Python projects.",
"Hardik uses scikit-learn and TensorFlow for machine learning projects in Python.",
"Hardik uses NLTK (Natural Language Toolkit) and spaCy for natural language processing tasks in Python.",
"Hardik uses async and await keywords with asyncio for asynchronous programming in Python, improving performance for I/O-bound tasks.",
"Hardik likes using Flask-RESTful and Django REST Framework for building RESTful APIs in Python.",
"Hardik uses pytest and unittest for writing and running tests in Python projects.",
"Hardik prefers using BeautifulSoup and Scrapy for web scraping tasks in Python.",
"Hardik uses pandas and NumPy for data analysis and manipulation in Python.",
"Hardik likes using Tkinter and PyQt for GUI development in Python.",
"Hardik uses pip for installing and managing dependencies and requirements.txt files to specify project dependencies.",
"Hardik prefers using BeautifulSoup and Scrapy for web scraping tasks in Python.",
"Hardik uses concurrent.futures and multiprocessing modules for concurrent programming in Python, optimizing CPU-bound tasks.",
"Hardik likes Flask-SQLAlchemy and Django ORM for database management in Python web applications.",
"Hardik has used Flask and FastAPI for building microservices in Python projects.",
"Hardik uses built-in file handling functions like open(), read(), write(), and pathlib module for file operations in Python.",
"Hardik likes using Flask-WTF and Django Forms for handling forms and user input validation in Python web applications.",
"Hardik uses Plotly and Dash for interactive data visualization in Python projects.",
"Hardik implements authentication using Flask-Login and Django authentication system for securing Python web applications.",
"Hardik uses Pillow and OpenCV for image processing tasks in Python.",
"Hardik optimizes database queries using Django ORM's select_related() and prefetch_related() methods for minimizing database hits.",
"Hardik likes using Flask-Mail and Django-REST-Knox for handling email sending and authentication tokens in Python web applications.",
"Hardik uses boto3 for AWS integration and google-cloud-python for Google Cloud Platform integration in Python projects.",
"Hardik uses Flask-Session and Django's session middleware for managing user sessions in Python web applications.",
"Hardik uses BeautifulSoup, Scrapy and Selenium for web scraping tasks in Python.",
"Hardik ensures data security by encrypting sensitive data using libraries like cryptography and implementing HTTPS for secure communication.",
"Hardik uses Fabric and Ansible for automated deployment and configuration management of Python web applications."
]

embeddings = model.encode(facts)

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query")
    query_embedding = model.encode(user_query)

    # Find the closest matching fact
    distances = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    closest_idx = np.argmax(distances)

    response = facts[closest_idx]
    return jsonify({"response": response})

@app.route("/keep-alive", methods=["GET"])
def keep_alive():
    return jsonify({"status": "alive"})

def keep_server_alive():
    with app.app_context():
        # Making a self-request to keep the server active
        response = app.test_client().get('/keep-alive')
        print(f"Keep-alive check: {response.json}")

if __name__ == "__main__":
    # Initialize APScheduler
    scheduler = BackgroundScheduler()
    # Schedule the keep-alive check every 5 minutes
    scheduler.add_job(keep_server_alive, 'interval', minutes=5)
    scheduler.start()

    try:
        app.run(host="0.0.0.0", port=5000)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        scheduler.shutdown()
