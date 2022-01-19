import implicit

import basic_collaborative_filtering as bcf
import kunn_collaborative_filtering as kunn
import test_model

def stockfunction(recommendation_list, product, model_kunn, X):

    for index in range(X):
        product_of_interest = product[index]
        corresponding_stockcode = list(model_kunn.products_map.keys())[list(model_kunn.products_map.values()).index(product_of_interest)]

        # top 10 recommendations
        X = 10
        topX = recommendation_list[index][0:X]
        recommendation_stock_codes = top_stock_codes(topX, model_kunn)

        print(f'index of interest: {product_of_interest}')
        print(f'corresponding stockcode: {corresponding_stockcode}')
        print(f'top {X} recommendations: {topX}')
        print(f'corresponding stockcodes: {recommendation_stock_codes}')
        print('')

def top_stock_codes(topX, model_kunn):
    stockcode_list = []
    for index in range(len(topX)):
        i = topX[index]
        stockcode = list(model_kunn.products_map.keys())[list(model_kunn.products_map.values()).index(i)]
        stockcode_list.append(stockcode)
    
    # should look at lowercase stockcodes 

    return stockcode_list
         