from flask import Flask, render_template, redirect, url_for


app = Flask(__name__)

def get_categories():
    # 模拟数据，接口以后再写
    return [
        {'name': 'test 1', 'description': 'category A'},
        {'name': 'test 2', 'description': 'category A'},
        {'name': 'test 3', 'description': 'category B'},
        {'name': 'test 4', 'description': 'category C'},
        {'name': 'test 5', 'description': 'category C'},
        {'name': 'test 6', 'description': 'category E'},
        {'name': 'test 7', 'description': 'category E'},
        {'name': 'test 8', 'description': 'category F'}
    ]

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/guide')
def guide():
    # 用数据库拉清单
    categories = get_categories()
    return render_template('guide.html', categories=categories)

@app.route('/identify')
def identify():
    return render_template('identify.html')

@app.route('/')
def index():
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
