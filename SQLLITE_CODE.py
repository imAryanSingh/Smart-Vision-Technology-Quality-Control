import sqlite3

# Function to create a database and a table
def create_database():
    conn = sqlite3.connect(r'C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\products.db')  # Update with your database path
    cursor = conn.cursor()
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ProductName TEXT,
            MRP TEXT,
            BrandName TEXT,
            SizeDetail TEXT,
            ManufacturingDate TEXT,
            ExpiryDate TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert product details into the database
def insert_product_details(product_name, mrp, brand_name, size_detail, manufacturing_date, expiry_date):
    conn = sqlite3.connect(r'C:\Users\aryan\OneDrive\Desktop\Flipkart_GRiD_Winner\products.db')  # Update with your database path
    cursor = conn.cursor()
    # Insert product details
    cursor.execute('''
        INSERT INTO Products (ProductName, MRP, BrandName, SizeDetail, ManufacturingDate, ExpiryDate)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (product_name, mrp, brand_name, size_detail, manufacturing_date, expiry_date))
    conn.commit()
    conn.close()

# Example usage:
if __name__ == "__main__":
    create_database()  # Create the database and table
    # Sample product details to insert
    sample_products = [
        ("Ariel", "99.99", "Ariel", "500g", "2023-01-01", "2024-01-01"),
        ("Fanta", "199.99", "Brand B", "1kg", "2022-12-15", "2023-12-15"),
        ("Lifebuoy Soap", "149.50", "Brand C", "250g", "2023-03-10", "2024-03-10")
    ]
    for product in sample_products:
        insert_product_details(*product)  # Unpack the tuple into function arguments
    print("Sample product details added to the database.")
