from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from faker import Faker

app = Flask(__name__)
fake = Faker()

# Generate a dataset with fake user profiles
def generate_dataset(n_samples):
    names = [fake.name() for _ in range(n_samples)]
    addresses = [fake.address() for _ in range(n_samples)]
    phones = [fake.phone_number() for _ in range(n_samples)]
    df = pd.DataFrame({'name': names, 'address': addresses, 'phone': phones})
    return df

# Load the dataset and train the classifier
df = generate_dataset(1000)
X = df[['name', 'address']]
y = df['phone']
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    # Get the input values and convert them to float
    name = float(request.form['name'])
    address = float(request.form['address'])
    
    # Generate a fake user profile using the classifier
    phone = clf.predict([[name, address]])[0]
    
    # Render the success page with the user profile
    return render_template('success.html', name=name, address=address, phone=phone)

if __name__ == '__main__':
    app.run(debug=True)
