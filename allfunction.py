import category_encoders as ce
# Encoder
def encoder(df): # cette fonction encode notre dataset et renvoie X_train,X_test,y_train,y_test
    # split X into training and testing sets
    X_train = df
    # encode variables with ordinal encoding
    col=[]
    for i in range(len(df.columns)):
        if df[df.columns[i]].dtypes=='O':
            col.append(df.columns[i])
    encoder = ce.OrdinalEncoder(cols=col)
    x_train = encoder.fit_transform(X_train)
    return x_train
    #----------------------------------------
