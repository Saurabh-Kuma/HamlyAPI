from flask import Flask, request
import pickle
import re

app = Flask(__name__)

# Load the model
tokeniserModel = pickle.load(open('cvMailTokeniser.pkl', 'rb'))
classifierModel = pickle.load(open('mailClassifier.pkl', 'rb'))

#email Cleaner Function
def emailCleaner(email):
    """This method accepts emails and returns absolutely cleaned emails by removing un-necessary headings"""
    
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
    #cleaning data
    cleanData = emailCleaner(data['text'])
    cleanedEmails = []
    cleanedEmails.append(cleanData)
    # Make prediction using the model loaded from disk
    tokenisedData = tokeniserModel.transform(cleanedEmails)
    prediction = classifierModel.predict(tokenisedData) 
    # Take the first value of prediction
    if(prediction[0]==0):
        return ("This Mail is Not Spam")
    return ("This Mail is Spam! Stay Alert")


if __name__ == '__main__':
    app.run(debug=True)
