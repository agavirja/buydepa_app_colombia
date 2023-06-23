def boxnumbermoney(number,money,label):
    style_boxnumber = """
        <style>
          .dashboard-box-boxnumber {
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
            padding: 20px;
            text-align: center;
            margin-bottom: 30px;
          }
          .number_boxnumber {
            color: #fff;
            display: inline-block;
          }
          .number_boxnumber h3 {
            font-size: 50px;
            color: #000;
          }
          .number_boxmoney {
            color: #fff;
            display: inline-block;
          }
          .number_boxmoney h3 {
            font-size: 20px;
            color: #000;
          }
          .number_boxnumber,
          .number_boxmoney {
            display: block; /* Cambia el valor de display a block para mostrarlos uno debajo del otro */
            margin-top: 10px; /* Agrega un margen superior para separar los elementos */
          }          
        </style>
    """
    html_boxnumber = f"""
        <!DOCTYPE html>
        <html>
          <head>
          {style_boxnumber}
          </head>
          <body>
            <div class="dashboard-box-boxnumber">
              <div class="number_boxnumber">
                <h3>{number}</h3>
              </div>
              <div class="number_boxmoney">
                <h3>{money}</h3>
              </div>
              <p>{label}</p>
            </div>
          </body>
        </html>    
    """
    return html_boxnumber

def boxnumberpercentage(number,percentage,label):
    style_boxnumberpercentage = """
        <style>
      .dashboard-box-boxnumberpercentage {
        background-color: #fff;
        border-radius: 5px;
        box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
        padding: 5px;
        text-align: center;
        margin-bottom: 10px;
      }

      .number_boxnumberpercentage {
        color: #fff;
        display: inline-block;
        margin-bottom: 0px;
      }

      .number_boxnumberpercentage h3 {
        font-size: 30px;
        color: #000;
        margin: 0px;
        padding: 0px;
      }

      .number_boxmoneypercentage {
        color: #fff;
        display: inline-block;
        margin-bottom: 0px;
        margin-left: 5px;
        margin-right: 5px;
      }

      .number_boxmoneypercentage h3 {
        font-size: 18px;
        color: #000;
        margin: 0px;
        padding: 0px;
      }

      .number_boxnumberpercentage,
      .number_boxmoneypercentage {
        display: block;
        margin-bottom: 5px;
      }

      p {
        margin: 0px;
        padding: 0px;
        margin-bottom: 5px;
        font-size: 14px;
      }
        </style>
    """
    html_boxnumberpercentage = f"""
        <!DOCTYPE html>
        <html>
          <head>
          {style_boxnumberpercentage}
          </head>
          <body>
            <div class="dashboard-box-boxnumberpercentage">
              <div class="number_boxnumberpercentage">
                <h3>{number}</h3>
              </div>
              <div class="number_boxmoneypercentage">
                <h3>{percentage}</h3>
              </div>
              <p>{label}</p>
            </div>
          </body>
        </html>    
    """
    return html_boxnumberpercentage

def boxkpi(number,label):
    style_boxnumber = """
        <style>
          .dashboard-box-boxkpi {
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 0 5px rgba(0, 0, 0, 0.3);
            padding: 20px;
            text-align: center;
            margin-bottom: 30px;
          }
          .number_boxkpi {
            color: #fff;
            display: inline-block;
          }
          .number_boxkpi h3 {
            font-size: 30px;
            color: #000;
          }
        </style>
    """
    html_boxnumber = f"""
        <!DOCTYPE html>
        <html>
          <head>
          {style_boxnumber}
          </head>
          <body>
            <div class="dashboard-box-boxkpi">
              <div class="number_boxkpi">
                <h3>{number}</h3>
              </div>
              <p>{label}</p>
            </div>
          </body>
        </html>    
    """
    return html_boxnumber

def table1(html,label):
    style = """
    <style>
            #tblStocks {
              font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
              border-collapse: collapse;
              width: 100%;
            }
            #tblStocks td, #tblStocks th {
              border: 1px solid #ddd;
              padding: 8px;
            }
            #tblStocks tr:nth-child(even){background-color: #f2f2f2;}
            #tblStocks tr:hover {background-color: #ddd;}
            #tblStocks th {
                padding-top: 12px;
                padding-bottom: 12px;
                text-align: center;
                background-color: #294c67;;
                color: white;
                font-size: 14px;
              }
            #tblStocks td {
                font-size: 12px;
            }
            .tabla {
              margin-bottom: 22px;
            }    
    </style>
    """
    texto = f"""
    <html>
    {style}
    <body>
        <table id="tblStocks" cellpadding="0" cellspacing="50" class="tabla">
        <tr>
            <th colspan="2">{label}</th>
        </tr>
        {html}
        </table>
    </body>
    </html>
    """
    return texto

def table2(html,label):
    style = """
    <style>
            #tblStocks {
              font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
              border-collapse: collapse;
              width: 100%;
            }
            #tblStocks td, #tblStocks th {
              border: 1px solid #ddd;
              padding: 8px;
            }
            #tblStocks tr:nth-child(even){background-color: #f2f2f2;}
            #tblStocks tr:hover {background-color: #ddd;}
            #tblStocks th {
                padding-top: 12px;
                padding-bottom: 12px;
                text-align: center;
                background-color: #294c67;;
                color: white;
                font-size: 14px;
              }
            #tblStocks td {
                font-size: 12px;
            }
            .tabla {
              margin-bottom: 22px;
            }    
    </style>
    """
    texto = f"""
    <html>
    {style}
    <body>
        <table id="tblStocks" cellpadding="0" cellspacing="50" class="tabla">
        <tr>
            <th colspan="4">{label}</th>
        </tr>
        {html}
        </table>
    </body>
    </html>
    """
    return texto

def table3(html,label1,label2,label3):
    style = """
    <style>
            #tblStocks {
              font-family: "Trebuchet MS", Arial, Helvetica, sans-serif;
              border-collapse: collapse;
              width: 100%;
            }
            #tblStocks td, #tblStocks th {
              border: 1px solid #ddd;
              padding: 8px;
            }
            #tblStocks tr:nth-child(even){background-color: #f2f2f2;}
            #tblStocks tr:hover {background-color: #ddd;}
            #tblStocks th {
                padding-top: 12px;
                padding-bottom: 12px;
                text-align: center;
                background-color: #294c67;;
                color: white;
                font-size: 14px;
              }
            #tblStocks td {
                font-size: 12px;
            }
            .tabla {
              margin-bottom: 22px;
            }    
    </style>
    """
    texto = f"""
    <html>
    {style}
    <body>
        <table id="tblStocks" cellpadding="0" cellspacing="50" class="tabla">
        <tr>
            <th>{label1}</th>
            <th>{label2}</th>
            <th>{label3}</th>
        </tr>
        {html}
        </table>
    </body>
    </html>
    """
    return texto

def timelineproperty(proceso):
    example = """
				<div class="swiper-slide">
				  <div class="timestamp"><span class="date">12.07.2019</span></div>
				  <div class="statusdark"><span>Work</span></div>
				</div>
				<div class="swiper-slide">
				  <div class="timestamp"><span class="date">12.07.2019</span></div>
				  <div class="statusdark"><span>Work</span></div>
				</div>
				<div class="swiper-slide">
                      <div class="timestamp"><span class="date">12.07.2019</span></div>
                      <div class="statuslight"><span>Work</span></div>
                </div>
                <div class="swiper-slide">
                  <div class="timestamp"><span class="date">12.07.2019</span></div>
                  <div class="statuslight"><span>Work</span></div>
                </div>
                <div class="swiper-slide">
                  <div class="timestamp"><span class="date">12.07.2019</span></div>
                  <div class="statuslight"><span>Work</span></div>
                </div>
                <div class="swiper-slide">
                  <div class="timestamp"><span class="date">12.07.2019</span></div>
                  <div class="statuslight"><span>Work</span></div>
                </div>
    """
    style = """
    	<style>
    		.time-line-box {
    			height: 100px; 
    			padding: 100px 0;
    			width: 100%;
                margin-bottom: 80px;
    		}
    		
    		.time-line-box .timeline {
    			list-style-type: none;
    			display: flex;
    			padding: 0;
    			text-align: center;
    		}
    		
    		.time-line-box .timestamp {
    			margin: auto;
                font-size: 16px;
    			margin-bottom: 20px;
    			padding: 0px 4px;
    			display: flex;
    			flex-direction: column;
    			align-items: center;
    		}
    		
    		.time-line-box .statuslight {
    			padding: 0px 10px;
    			display: flex;
    			justify-content: center;
    			align-items: center;
    			border-top: 3px solid #455EFC;
    			height: 50px;
    			position: relative;
    			transition: all 200ms ease-in;
    		}
    		
    		.time-line-box .statuslight span {
    			padding-top: 8px;
    		}
    		
    		.time-line-box .statuslight span:before {
                content: '';
    			width: 40px;
    			height: 40px;
    			border-radius: 20px;
                background-color: #FFFFFF;
    			border: 2px solid #455EFC;
    			position: absolute;
    			left: 50%;
    			top: 0%;
    			-webkit-transform: translate(-50%, -50%);
    			-ms-transform: translate(-50%, -50%);
    			transform: translate(-50%, -50%);
    			transition: all 200ms ease-in;
    		}
    		.time-line-box .statusdark {
    			padding: 0px 10px;
    			display: flex;
    			justify-content: center;
    			align-items: center;
    			border-top: 3px solid #455EFC;
    			height: 50px;
    			position: relative;
    			transition: all 200ms ease-in;
    		}
    		
    		.time-line-box .statusdark span {
    			padding-top: 8px;
    		}
    		
    		.time-line-box .statusdark span:before {
    			content: '';
    			width: 40px;
    			height: 40px;
    			border-radius: 20px;
                background-color: #455EFC;
    			border: 2px solid #455EFC;
    			position: absolute;
    			left: 50%;
    			top: 0%;
    			-webkit-transform: translate(-50%, -50%);
    			-ms-transform: translate(-50%, -50%);
    			transform: translate(-50%, -50%);
    			transition: all 200ms ease-in;
    		}
    
    		.swiper-container {
    			width: 95%; 
    			margin: auto;
    			overflow-y: auto;
    		}
    		
    		.swiper-wrapper{
    			display: inline-flex;
    			flex-direction: row;
    			overflow-y:auto;
    			justify-content: center;
    		}
    		
    		.swiper-container::-webkit-scrollbar-track{
    		   background:#a8a8a8b6;
    		}
    		
    		.swiper-container::-webkit-scrollbar{
    			height: 2px;
    		}
    		
    		.swiper-container::-webkit-scrollbar-thumb{
    		   background: #4F4F4F !important;
    		}
    		
    		.swiper-slide {
    			text-align: center;
    			font-size: 16px;
    			width: 150px;
    			height: 150px;
    			position: relative;
    		}
            .time-line-box .statuslight span,
            .time-line-box .statusdark span {
              margin-top: 50px;
            }
    	</style>
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    	<meta charset="utf-8">
    	<meta name="viewport" content="width=device-width, initial-scale=1">
    	<link rel="stylesheet" href="https://unpkg.com/swiper/swiper-bundle.min.css" />
        {style}
    </head>
    <body>
    	<section class="time-line-box">
    		<div class="swiper-container text-center"> 
    			<div class="swiper-wrapper">
                {proceso}
                </div>
            <div class="swiper-pagination"></div>
            </div>
           </section>
    </body>
    </html>
    """
    return html

def html_estado_propiedad(estado):
    style = """
        <style>
          .estadonovendido {
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: center;
            background-color: #ED6E3F;
            color: white;
            font-size: 16px;
          }
          .estadovendido {
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: center;
            background-color: #008f39;
            color: white;
            font-size: 16px;
        </style>
    """
    
    html = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        {style}
      </head>
      <body>
        {estado}
      </body>
    </html>
    """
    return html

def imgpropertylist(imagenes):
    css_format = """
        <style>
          .property-card-left {
            width: 100%;
            text-align: center;
            display: inline-block;
            margin: 0px auto;
          }

          .property-block {
            width:32%;
            background-color: white;
            border: 1px solid gray;
            box-shadow: 2px 2px 2px gray;
            padding: 3px;
            margin-bottom: 10px; 
      	    display: inline-block;
      	    float: left;
            margin-right: 10px; 
          }

          .property-image{
            flex: 1;
          }
     
        
          .mi-imagen {
            max-width: 100%;
            width: 100%;
            height: 250px;
            object-fit: cover;
            margin-bottom: 10px; 
          }
        </style>
    """

    texto = f"""
        <!DOCTYPE html>
        <html>
          <head>
          {css_format}
          </head>
          <body>
        <div class="property-card-left">
        {imagenes}
        </div>
          </body>
        </html>
        """
    return texto