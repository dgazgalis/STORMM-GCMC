#include <string>
#include <vector>
#include "../../src/copyright.h"
#include "../../src/Constants/behavior.h"
#include "../../src/DataTypes/common_types.h"
#include "../../src/DataTypes/stormm_vector_types.h"
#include "../../src/Namelists/command_line_parser.h"
#include "../../src/Namelists/input.h"
#include "../../src/Namelists/namelist_emulator.h"
#include "../../src/Namelists/namelist_enumerators.h"
#include "../../src/Namelists/namelist_inventory.h"
#include "../../src/Parsing/parsing_enumerators.h"
#include "../../src/Parsing/textfile.h"
#include "../../src/Reporting/error_format.h"
#include "../../src/Reporting/help_messages.h"

using stormm::constants::CaseSensitivity;
using stormm::constants::ExceptionResponse;
using stormm::display::displayNamelistHelp;
using stormm::errors::rtErr;
using stormm::errors::terminalFormat;
using stormm::parse::TextFile;
using stormm::parse::WrapTextSearch;
using stormm::namelist::CommandLineParser;
using stormm::namelist::InputStatus;
using stormm::namelist::NamelistEmulator;
using stormm::namelist::NamelistToken;
using stormm::namelist::NamelistType;
using stormm::namelist::readNamelist;

//-------------------------------------------------------------------------------------------------
NamelistEmulator starInput(const TextFile &input_ascii_text, int *start_line, bool *found_nml,
                           ExceptionResponse policy = ExceptionResponse::DIE,
                           WrapTextSearch wrap = WrapTextSearch::NO) {

  // Create the namelist object
  NamelistEmulator result("star", CaseSensitivity::YES, ExceptionResponse::DIE, "Parameters for "
                          "a star in the sky");

  // Configure its keywords
  result.addKeyword("planets", NamelistType::INTEGER, std::to_string(8));
  result.addKeyword("mass", NamelistType::REAL, std::to_string(1.0));
  result.addKeyword("brightness", NamelistType::REAL, std::to_string(-26.74));
  result.addKeyword("name", NamelistType::STRING, std::string(""));
  result.addKeyword("constellation", NamelistType::STRING, std::string(""));

  // Configure help messages for each keyword
  result.addHelp("planets", "The number of planets known to orbit the star");
  result.addHelp("mass", "Estimated mass of the star, in units of solar masses");
  result.addHelp("brightness", "Apparent magnitude of the star, as observed from Earth");
  result.addHelp("name", "Common name given to the star");
  result.addHelp("constellation", "Name of the constellation in which the star appears, or the "
                 "constellation giving a direction in which to find the star");
  
  // With the namelist configured, run the input text through it to collect user directives
  *start_line = readNamelist(input_ascii_text, &result, *start_line, wrap,
                             input_ascii_text.getLineCount(), found_nml);
  return result; 
}

//-------------------------------------------------------------------------------------------------
// The refined form of user input, collecting information about a star in the night sky.
//-------------------------------------------------------------------------------------------------
class StarControls {
public:
  
  // The constructor will require the contents of an input file, already converted to a text
  // class object.
  //
  // Arguments:
  //   input_ascii_text:  The text file contents as read into main memory
  //   start_line:        The first line at which to begin searching the input text
  //   found_nml:         Indication of whether the namelist was found
  StarControls(const TextFile &input_ascii_text, int *start_line, bool *found_nml);

  // Get the number of planets in orbit
  int getPlanetCount() const;

  // Get the mass in units of solar masses
  double getMass() const;

  // Get the brightness (apparent magnitude)
  double getBrightness() const;

  // Get the star's common name
  const std::string& getCommonName() const;

  // Get the star's constellation
  const std::string& getConstellation() const;

  // Get a const reference to the original namelist transcript
  const NamelistEmulator& getTranscript() const;
  
private:
  int planet_count;           // Number of planets in orbit
  double mass;                // Mass in units of solar masses
  double brightness;          // Brightness expressed in apparent magnitude
  std::string common_name;    // The name of the star
  std::string constellation;  // Name of a constellation that the star may be part of

  // Keep a transcript of the user input, for reporting purposes
  NamelistEmulator transcript;
};

//-------------------------------------------------------------------------------------------------
StarControls::StarControls(const TextFile &input_ascii_text, int *start_line, bool *found_nml) :
    planet_count{8}, mass{1.0}, brightness{-26.74}, common_name{}, constellation{},
    transcript{"star"}
{
  NamelistEmulator t_nml = starInput(input_ascii_text, start_line, found_nml);
  planet_count = t_nml.getIntValue("planets");
  mass = t_nml.getRealValue("mass");

  // We can also use the assignVariable() method to transfer data from the namelist
  t_nml.assignVariable(&brightness, "brightness");

  // Some keywords may not have been specified.  We can check whether the keyword has been entered
  // by the user, and transfer the data if it has.
  if (t_nml.getKeywordStatus("name") != InputStatus::MISSING) {
    common_name = t_nml.getStringValue("name");
  }
  
  // An alternative is to use assignVariable() again, which will do the check automatically.
  t_nml.assignVariable(&constellation, "constellation");
}

//-------------------------------------------------------------------------------------------------
int StarControls::getPlanetCount() const {
  return planet_count;
}

//-------------------------------------------------------------------------------------------------
double StarControls::getMass() const {
  return mass;
}

//-------------------------------------------------------------------------------------------------
double StarControls::getBrightness() const {
  return brightness;
}

//-------------------------------------------------------------------------------------------------
const std::string& StarControls::getCommonName() const {
  return common_name;
}

//-------------------------------------------------------------------------------------------------
const std::string& StarControls::getConstellation() const {
  return constellation;
}

//-------------------------------------------------------------------------------------------------
const NamelistEmulator& StarControls::getTranscript() const {
  return transcript;
}

//-------------------------------------------------------------------------------------------------
// main
//-------------------------------------------------------------------------------------------------
int main(int argc, const char* argv[]) {

  // Configure the command line parser and the namelist within it
  CommandLineParser clip("Tutorial_III", "An exercise in creating custom user control blocks with "
                         "the native STORMM documentation system");
  clip.addStandardApplicationInputs("-i");
  NamelistEmulator *cmd_nml = clip.getNamelistPointer();
  cmd_nml->addKeyword("-rating", NamelistType::INTEGER, std::string(""));
  cmd_nml->addHelp("-rating", "Rate this tutorial on a scale of 1-10.");
  const std::vector<NamelistToken> tutorial_specific_namelists = {
    NamelistToken(std::string("&star"), starInput)
  };
  const std::vector<std::string> my_namelist_names = { "star" };
  clip.addControlBlocks(my_namelist_names);
  clip.addCustomNamelists(tutorial_specific_namelists);
  if (displayNamelistHelp(argc, argv, my_namelist_names, tutorial_specific_namelists) &&
      clip.doesProgramExitOnHelp()) {
    return 0;
  }
  
  // Parse the command line data
  clip.parseUserInput(argc, argv);

  // If the user has provided a rating for the tutorial, respond
  int user_rating = 0;
  cmd_nml->assignVariable(&user_rating, "-rating");
  if (user_rating > 0) {
    printf("Thank you for rating this tutorial!\n");
  }
  std::string rating_response;
  switch (user_rating) {
  case 1:
  case 2:
    rating_response = terminalFormat("We're sorry that your experience wasn't so good.  Feel free "
                                     "to send feedback to David Cerutti "
                                     "(david.cerutti@psivant.com).\n");
    break;
  case 3:
  case 4:
  case 5:
  case 6:
    rating_response = terminalFormat("Tutorials for STORMM will continue to develop throughout "
                                     "the course of the project.  We will continue to improve "
                                     "this one and others.\n");
    break;
  case 7:
  case 8:
    rating_response = terminalFormat("Sounds like you've had a reasonably positive experience for "
                                     "a tutorial of a new code base.  Hopefully you've found "
                                     "motivation to do more.\n");
    break;
  case 9:
  case 10:
    rating_response = terminalFormat("Stupendous!  We're very pleased that you have liked the "
                                     "tutorial, and look forward to anything you might want to "
                                     "contribute.  Contact the lead developer, David Cerutti, at "
                                     "david.cerutti@psivant.com to get more involved.\n");
    break;
  default:
    break;
  }
  printf("%s\n", rating_response.c_str());
  
  // Read the input file
  std::string input_file_name;
  cmd_nml->assignVariable(&input_file_name, "-i");
  const TextFile input_ascii_text(input_file_name);
  std::vector<StarControls> night_sky;
  int start_line = 0;
  bool found_nml = false;
  do {
    night_sky.emplace_back(input_ascii_text, &start_line, &found_nml);
    if (found_nml == false) {
      night_sky.pop_back();
    }
  } while (found_nml);

  // Report on the input detected
  printf("There are %zu stellar entries in %s.\n\n", night_sky.size(),
         input_file_name.c_str());
  printf("    Star Name      Mass  Brightness Planets   Constellation\n");
  printf("----------------- ------ ---------- ------- -----------------\n");
  for (size_t i = 0; i < night_sky.size(); i++) {
    printf("%17.17s %6.2lf %10.4lf %7d %17.17s\n", night_sky[i].getCommonName().c_str(),
           night_sky[i].getMass(), night_sky[i].getBrightness(), night_sky[i].getPlanetCount(),
           night_sky[i].getConstellation().c_str());
  }

  // Return success
  return 0;
}
