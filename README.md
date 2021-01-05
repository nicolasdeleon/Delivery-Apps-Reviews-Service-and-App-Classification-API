# Base API for review classification

Idea came from reviews: 

[Pedidos Ya App Reviews](https://play.google.com/store/apps/details?id=com.pedidosya&hl=es_419&showAllReviews=true)



## Getting Started

- Download model binary file from [link](https://drive.google.com/file/d/1jzTaBFqzEebsQFAQGlmYOg_j8QgouOCV/view?usp=sharing)

- Download tokenizer from [link](https://drive.google.com/file/d/1r95yFbJDaRGSYlIhPNhc3M4pMkXG3o4W/view?usp=sharing)

- Unzip both folders in main directory

- Entry point src/api

## Example

```
{
  "response": {
    "prediction": {
      "input": "El servicio muy lento. El celular por otro lado se traba todo el tiempo", 
      "prediction": "App and Service", 
      "prediction values": {
        "app": -2.71728515625, 
        "app and service": 5.459717273712158, 
        "other": -2.654169797897339, 
        "service": 0.3330422639846802
      }
    }
  }
}
```
