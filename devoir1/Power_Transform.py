'''
LISEZ ATTENTIVEMENT CE COMMENTAIRE 

Pour que nos scripts d'autoévaluation fonctionnent correctement et vous attribuent une note pour votre devoir 
(au lieu d'un zéro) respectez A LA LETTRE les directives indiquées en dessous :

   Après avoir modifier ce source entre les markeurs indiqués en bas, executer le code depuis la console IPython comme suit :

       %run Power_Transform.py -i image_source.jpg -o image_resultat.jpg 

N'oubliez pas de COMMENTER votre code (20% de la note)
N'oubliez pas de POUSSER votre solution (ce fichier modifié avec votre code) vers votre serveur GitLab (voir le document du devoir)! 
'''

import sys, getopt
from PIL import Image 
import numpy as np

'''
fonction Power_Transform()
pour effectuer une transformation ponctuelle utilisant la loi de la puissance sur une image. 

paramètres :
   inputfile  : nom de fichier de l'image source
   outputfile : nom de fichier de l'image résultat
   gamma      : valeur gamma à appliquer à l'image  
'''
def Power_Transform(inputfile, outputfile, gamma = 1.0):

   # lecture de l'image et conversion RGB -> niveau de gris
   img  = Image.open(inputfile).convert('L')   

   # formatage de l'image sous forme de tableau numpy (ici -> matrice 2D)
   data = np.array(img)  
                       
   ##################### VOTRE CODE COMMENCE ICI (ne rien modifier en dessus de cette ligne) ######################



   ##################### VOTRE CODE FINI ICI  (ne rien modifier en dessous de cette ligne) ########################

   # préparation de l'image de sortie à partir du tableau numpy (i.e la matrice associé à l'image)
   result = Image.fromarray(data)

   # sauvegarde de l'image résultat
   result.save(outputfile)

   #retour de la fonction
   return;
 
'''
fonction main du programme :
cette fonction parse la ligne de commande et récupère les arguments pour la fonction
GrayLevelModification(...) (i.e. les noms des fichiers en entrée et en sortie)
'''
def main(argv):

   inputfile = ''
   outputfile = ''
   gamma = 1.0

   try:
      opts, args = getopt.getopt(argv,"hi:o:g:",["ifile=","ofile=","gamma="])
   except getopt.GetoptError:
      print('Power_Transform.py -i <inputfile> -o <outputfile> -g <gamma>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('Power_Transform.py -i <inputfile> -o <outputfile> -g <gamma>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      elif opt in ("-g", "--gamma"):
         gamma = arg

   Power_Transform(inputfile, outputfile, gamma = gamma)

   print('program terminated')
   return;
   
main(sys.argv[1:])
'''
fin du code 
'''

