# import array
# import random
# import sys

# class PGMWriter():
#     def __init__(self, map, contain_wall_cylinders, filename):
#         self.map = map
#         self.rows = len(map)
#         self.cols = len(map[0]) + contain_wall_cylinders
#         self.contain_wall_cylinders = contain_wall_cylinders
#         self.filename = filename

#     def __call__(self):

#         # define the width  (columns) and height (rows) of your image
#         width = self.rows
#         height = self.cols + 3

#         buff=array.array('B')

#         # add 3 extra columns of open space at the end
#         for r in range(self.rows):
#             buff.append(255)
#             buff.append(255)
#             buff.append(255)

#         # write the actual obstacles
#         for c in range(self.cols - 1, -1, -1):
#             for r in range(self.rows):
#                 # add the containment wall
#                 if c < self.contain_wall_cylinders:
#                     if r == 0 or r == self.rows - 1:
#                         buff.append(0)
#                     elif c == 0:
#                         buff.append(0)
#                     else:
#                         buff.append(255)

#                 elif self.map[r][c - self.contain_wall_cylinders] == 1:
#                     buff.append(0)
#                 else:
#                     buff.append(255)


#         # open file for writing
#         try:
#             fout=open(self.filename, 'wb')
#         except IOError as er:
#             sys.exit()


#         # define PGM Header
#         pgm_header = 'P5' + '\n' + str(width) + '  ' + str(height) + '  ' + str(255) + '\n'

#         # write the header to the file
#         fout.write(pgm_header)

#         # write the data to the file 
#         buff.tofile(fout)

#         # close the file
#         fout.close()
import array
import sys

class PGMWriter():
    def __init__(self, map, contain_wall_cylinders, filename):
        self.map = map
        self.rows = len(map)
        self.cols = len(map[0]) + contain_wall_cylinders
        self.contain_wall_cylinders = contain_wall_cylinders
        self.filename = filename

    def __call__(self):

        # Define the width (columns) and height (rows) of your image
        width = self.rows
        height = self.cols + 3

        buff = array.array('B')

        # Add 3 extra columns of open space at the end
        for r in range(self.rows):
            buff.extend([255, 255, 255])

        # Write the actual obstacles
        for c in range(self.cols - 1, -1, -1):
            for r in range(self.rows):
                # Add the containment wall
                if c < self.contain_wall_cylinders:
                    if r == 0 or r == self.rows - 1:
                        buff.append(0)
                    elif c == 0:
                        buff.append(0)
                    else:
                        buff.append(255)

                elif self.map[r][c - self.contain_wall_cylinders] == 1:
                    buff.append(0)
                else:
                    buff.append(255)

        # Open file for writing
        try:
            fout = open(self.filename, 'wb')
        except IOError as er:
            print("I/O error({0}): {1}".format(er.errno, er.strerror))
            sys.exit()

        # Define PGM Header
        pgm_header = 'P5\n' + str(width) + ' ' + str(height) + ' 255\n'

        # Write the header to the file
        fout.write(pgm_header.encode('ascii'))

        # Write the data to the file 
        buff.tofile(fout)

        # Close the file
        fout.close()

# Example usage
# map = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
# writer = PGMWriter(map, 1, 'output.pgm')
# writer()

