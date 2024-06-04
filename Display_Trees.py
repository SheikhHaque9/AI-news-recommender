import tkinter as tk
from tkinter import ttk
from Recommend_Trees import *
from News_DB import *


class NewsApp:
    def __init__(self, root, dataframe):
        self.root = root
        self.df = dataframe
        self.create_widgets()
        self.liked_articles = []
        self.recommender = ArticleRecommenderDT(self.df)

    def create_widgets(self):
        # Left side section for navigation buttons
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.article_list_btn = ttk.Button(nav_frame, text="Article List", command=self.show_articles)
        self.article_list_btn.pack(fill=tk.X, pady=5)

        self.liked_articles_btn = ttk.Button(nav_frame, text="Liked Articles", command=self.show_liked_articles)
        self.liked_articles_btn.pack(fill=tk.X, pady=5)

        self.recommended_articles_btn = ttk.Button(nav_frame, text="Recommended Articles",
                                                   command=self.show_recommended_articles)
        self.recommended_articles_btn.pack(fill=tk.X, pady=5)

        # Middle section for listbox and like button
        self.list_frame = tk.Frame(self.root)
        self.list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.listbox = tk.Listbox(self.list_frame)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind('<<ListboxSelect>>', self.show_article_content)

        self.like_frame = tk.Frame(self.root)
        self.like_frame.pack(side=tk.LEFT, padx=20)

        self.like_button = ttk.Button(self.like_frame, text="Like >>", command=self.like_article)
        self.like_button.pack()

        # Right side Text widget for article content
        self.article_content = tk.Text(self.root, wrap=tk.WORD)
        self.article_content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Default view
        self.show_articles()

    def show_articles(self):
        # Show the 'Like' button
        self.like_button.pack()

        self.listbox.delete(0, tk.END)
        for _, row in self.df.iterrows():
            self.listbox.insert(tk.END, row['title'])
        self.article_content.delete(1.0, tk.END)

    def show_liked_articles(self):
        # Hide the 'Like' button
        self.like_button.pack_forget()

        self.listbox.delete(0, tk.END)
        for item in self.liked_articles:
            self.listbox.insert(tk.END, item)
        self.article_content.delete(1.0, tk.END)

    def like_article(self):
        selected_indices = self.listbox.curselection()
        if selected_indices:
            article_title = self.df.iloc[selected_indices[0]]['title']
            if article_title not in self.liked_articles:
                self.liked_articles.append(article_title)

    def show_recommended_articles(self):
        # Hide the 'Like' button
        self.like_button.pack_forget()

        # Get cleaned content of liked articles for recommendation
        liked_articles_content = [self.df[self.df['title'] == title]['cleaned_content'].iloc[0] for title in
                                  self.liked_articles]

        if not liked_articles_content:
            self.listbox.delete(0, tk.END)
            self.listbox.insert(tk.END, "No liked articles to make recommendations.")
            return

        recommended_titles = self.recommender.recommend(liked_articles_content)

        self.listbox.delete(0, tk.END)
        for title in recommended_titles:
            self.listbox.insert(tk.END, title)

        self.article_content.delete(1.0, tk.END)

    def show_article_content(self, evt):
        selected_index = self.listbox.curselection()
        if selected_index:
            description = self.df.iloc[selected_index[0]]['content']
            self.article_content.delete(1.0, tk.END)
            self.article_content.insert(tk.END, description)


root = tk.Tk()
root.geometry("1000x500")
app = NewsApp(root, df)
root.mainloop()