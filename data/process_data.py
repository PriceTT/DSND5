import sys
import pandas as pd
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''    Create dataframe for messages and categories data.
    Parameters:
    messages_filepath (string): location msgs csv
    categories_filepath (string): location categories csv
       
    Returns:
    messages_df (Dataframe): msges dataframe  
    categories_df (Dataframe): categories dataframe 
    '''
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    return messages_df, categories_df


def clean_data(messages, categories):
    '''    Create a clean, combined dataframe of messages and category dummy variables.
    Parameters:
    messages (Dataframe): msges dataframe  
    categories (Dataframe): categories dataframe 
    
    Returns:
    df (Dataframe): cleaned and merged dataframe  
    '''
    # Merge datasets
    df = messages.merge(categories, on='id')
 
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.str[:-2]
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
     
    # child alone is empty so drop
    categories.drop('child_alone', axis = 1, inplace = True)
        
    # drop the original categories column from `df`
    df.drop(['original','categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df =  pd.concat([df , categories], axis=1)

    # Clean 'related' values
    df = df[df['related'] != 2]
    
    df.dropna(axis=0,inplace=True) 

    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    '''    Save dataframe to database in 'messages' table. Replace any existing data.
    Parameters:
    df (Dataframe): cleaned dataframe  
    database_filename (string): name of database 
    
    Returns:
     
    
    '''
    conn = sqlite3.connect(database_filename)
    df.to_sql('messages', con=conn, if_exists='replace', index=False)
    conn.commit()
    conn.close() 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages_df, categories_df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages_df, categories_df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()