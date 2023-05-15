from flask import Flask, request
import pickle
import re

app = Flask(__name__)

# Load the model
model1 = pickle.load(open('cvMailTokeniser.pkl', 'rb'))
model2 = pickle.load(open('mailClassifier.pkl', 'rb'))


def emailCleaner(email):
    """This method accepts emails and returns absolutely cleaned emails by removing un-necessary headings"""
    
    # printing un-processed email
    # print("Email is: ",email)

    # converting emial into lower case
    emailsInLower = email.lower()

    # removing "Subject: " heading
    emailWithoutSubject = emailsInLower.replace("subject: ","")

    # removing "re : " (The reply mark)
    emailWithoutReplyMark = emailWithoutSubject.replace("re : ","")

    # removing all the punctuation marks
    emailWithoutPunctuation = re.sub(r"[^\w\s]" , "" , emailWithoutReplyMark)

    # removing extra whitespaces
    cleanedEmail = re.sub(r'\s{2,}', ' ', emailWithoutPunctuation.strip())
    return cleanedEmail


@app.route('/predict_mail', methods=['POST'])
def predict_mail():
    # Get the data from the POST request
    data = request.get_json(force=True)
    print(data)
    #cleaning data
    cleanData = emailCleaner(data['text'])
    # [np.array(data['text'])]
    cleanedEmails = []
    cleanedEmails.append(cleanData)
    # Make prediction using the model loaded from disk
    transformed = model1.transform(cleanedEmails)
    prediction = model2.predict(transformed) 
    # Take the first value of prediction
    if(prediction[0]==0):
        return (cleanData+ ": This Mail is Not Spam")
    return (cleanData+ ": This Mail is Spam! Stay Alert")


if __name__ == '__main__':
    app.run(debug=True)
