/* * * * * * * * * * * * * * * * * * * * * * * * * * *\
* v1.5   Kelbaz Artificial Intellingence    By Kelbaz *
*                                                     *
*        @@@@@@@@@@@@@@   @@,              @*`'@      *
*        @@@@@@@@@@@@'    @@@@,            @. ,@      *
*     ,@@ `@@@@@@@@'      @'  `@,          @@@@@      *
*     @@@  :@@@@@'        @.  ,@@@,        @@@@@      *
*     `@@ ,@@@@' @@,      @@@@@`@@@@,      @@@@@      *
*        @@@@'   @@@@,    @@@@@ `@@@@@,    @@@@@      *
*        @@'     @@@@@@   @@@@@   `@@@@@   @@@@@      *
*                                                     *
\* * * * * * * * * * * * * * * * * * * * * * * * * * */

import * as activation from "./activations.js";
import * as network from "./network.js";

if (typeof window !== "undefined") {
  window.K_AI = {
    activation,
    network,
  }
} else {
  module.exports = {
    activation,
    network,
  }
}