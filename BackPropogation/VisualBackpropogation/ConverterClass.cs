using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Data;

namespace VisualBackPropogation
{
    class ConverterClass : IValueConverter
    {

        //Convert a math value to a new math value
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            String[] string_list = ((string)parameter).Split(';');
            double x = (double)value;
            double y;
            for (int i = 0; i < string_list.Length; i += 2)
            {
                y = double.Parse((string)string_list[i + 1]);
                switch (string_list[i])
                {
                    case "+": 
                        x += y;
                        break;

                    case "-":
                        x -= y;
                        break;
                    case "*": 
                        x *= y;
                        break;
                    case "/": 
                        x /= y;
                        break;
                    case "Multiply": 
                        x *= y;
                        break;
                    case "Add": 
                        x += y;
                        break;
                    case "Divide": 
                        x /= y;
                        break;
                    case "Subtract": 
                        x -= y;
                        break;
                    default: throw new Exception("invalid logic");
                }
            }
            return x;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return true;
        }

    }
}
