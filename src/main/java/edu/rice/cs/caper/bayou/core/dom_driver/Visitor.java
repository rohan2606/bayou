/*
Copyright 2017 Rice University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package edu.rice.cs.caper.bayou.core.dom_driver;

import com.google.common.collect.Multiset;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import edu.rice.cs.caper.bayou.core.dsl.DASTNode;
import edu.rice.cs.caper.bayou.core.dsl.DSubTree;
import edu.rice.cs.caper.bayou.core.dsl.Sequence;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.eclipse.jdt.core.dom.*;
import org.eclipse.jdt.core.dom.SingleVariableDeclaration;
import org.eclipse.jdt.core.dom.VariableDeclaration;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.stream.Collectors;

public class Visitor extends ASTVisitor {

    public final CompilationUnit unit;
    public final Options options;
    private final JSONOutput _js;

    public List<MethodDeclaration> allMethods;
    public List<FieldDeclaration> allTypes;

    // call stack during driver execution
    public final Stack<MethodDeclaration> callStack = new Stack<>();

    class JSONOutput {
        List<JSONOutputWrapper> programs;

        JSONOutput() {
            this.programs = new ArrayList<>();
        }
    }

    class JSONOutputWrapper {

        //Identifiers
        String file;
        String method;
        String body;

        // Output
        DSubTree ast;

        // Evidence inputs
        List<Sequence> sequences;

        // New Evidences Types
        String returnType;
        List<String> formalParam;
        List<String> classTypes;



        public JSONOutputWrapper(String methodName, String body, DSubTree ast, List<Sequence> sequences,  String returnType, List<String> formalParam,
        List<String> classTypes ) {

            this.file = options.file;
            this.method = methodName;
            this.body = body;

            this.ast = ast;

            this.sequences = sequences;

            this.returnType = returnType;
            this.formalParam = formalParam;
            this.classTypes = classTypes;


        }
    }

    public Visitor(CompilationUnit unit, Options options) throws FileNotFoundException {
        this.unit = unit;
        this.options = options;

        _js = new JSONOutput();
        allMethods = new ArrayList<>();
        allTypes = new ArrayList<>();
    }

    @Override
    public boolean visit(TypeDeclaration clazz) {
        if (clazz.isInterface())
            return false;
        List<TypeDeclaration> classes = new ArrayList<>();
        classes.addAll(Arrays.asList(clazz.getTypes()));
        classes.add(clazz);

        for (TypeDeclaration cls : classes){
            allMethods.addAll(Arrays.asList(cls.getMethods()));
            allTypes.addAll(Arrays.asList(cls.getFields()));
          }

        List<MethodDeclaration> constructors = allMethods.stream().filter(m -> m.isConstructor()).collect(Collectors.toList());
        List<MethodDeclaration> publicMethods = allMethods.stream().filter(m -> !m.isConstructor() && Modifier.isPublic(m.getModifiers())).collect(Collectors.toList());

        // synchronized lists
        List<DSubTree> asts = new ArrayList<>();
        List<String> methodNames = new ArrayList<>();
        List<String> returnTypes = new ArrayList<>();
        List<String> bodys = new ArrayList<>();
        List<List<String>> formalParams = new ArrayList<>();
        List<String> classTypes = new ArrayList<>();



        Collections.shuffle(allTypes);
        int maxClass = 10;
        int i = 0;
        for (FieldDeclaration type : allTypes){
          classTypes.add(type.getType().toString());
          i += 1;
          if (i>maxClass)
            break;
        }

        Collections.shuffle(constructors);
        Collections.shuffle(publicMethods);
        int maxSamples =  10;
        if (!constructors.isEmpty() && !publicMethods.isEmpty()) {

            i = 0;
            for (MethodDeclaration c : constructors){
              if (i>maxSamples)
                break;
              callStack.push(c);
              DSubTree ast = new DOMMethodDeclaration(c, this).handle();
              callStack.pop();
              if (ast.isValid()) {
                  asts.add(ast);
                  methodNames.add(c.getName().getIdentifier() + "@" + getLineNumber(c));
                  returnTypes.add(getReturnType(c));
                  bodys.add(c.toString());
                  formalParams.add(getFormalParams(c));
              }
              i += 1;
            }

            i = 0;
            for (MethodDeclaration m : publicMethods) {
                if (i>maxSamples)
                  break;
                callStack.push(m);
                DSubTree ast = new DOMMethodDeclaration(m, this).handle();
                callStack.pop();
                if (ast.isValid()) {
                    asts.add(ast);
                    methodNames.add(m.getName().getIdentifier() + "@" + getLineNumber(m));
                    returnTypes.add(getReturnType(m));
                    bodys.add(m.toString());
                    formalParams.add(getFormalParams(m));
                }
                i += 1;
            }
        } else if (!constructors.isEmpty()) { // no public methods, only constructor
            i = 0;
            for (MethodDeclaration c : constructors) {
                if (i>maxSamples)
                  break;
                callStack.push(c);
                DSubTree ast = new DOMMethodDeclaration(c, this).handle();
                callStack.pop();
                if (ast.isValid()) {
                    asts.add(ast);
                    methodNames.add(c.getName().getIdentifier() + "@" + getLineNumber(c));
                    returnTypes.add(getReturnType(c));
                    bodys.add(c.toString());
                    formalParams.add(getFormalParams(c));
                }
                i += 1;
            }
        } else if (!publicMethods.isEmpty()) { // no constructors, methods executed typically through Android callbacks
            i = 0;
            for (MethodDeclaration m : publicMethods) {
                if (i>maxSamples)
                  break;
                callStack.push(m);
                DSubTree ast = new DOMMethodDeclaration(m, this).handle();
                callStack.pop();
                if (ast.isValid()) {
                    asts.add(ast);
                    methodNames.add(m.getName().getIdentifier() + "@" + getLineNumber(m));
                    returnTypes.add(getReturnType(m));
                    bodys.add(m.toString());
                    formalParams.add(getFormalParams(m));
                }
                i += 1;
            }
        }



        for (i = 0; i < asts.size(); i++) {
          DSubTree ast = asts.get(i);
          String methodName = methodNames.get(i);
          String returnType = returnTypes.get(i);
          List<String> formalParam = formalParams.get(i);
          String body = bodys.get(i);

          List<Sequence> sequences = new ArrayList<>();
          sequences.add(new Sequence());
          try {
              ast.updateSequences(sequences, options.MAX_SEQS, options.MAX_SEQ_LENGTH);
              List<Sequence> uniqSequences = new ArrayList<>(new HashSet<>(sequences));
              if (okToPrintAST(uniqSequences)){
                addToJson(methodName, body, ast, uniqSequences, returnType, formalParam, classTypes);
              }
          } catch (DASTNode.TooManySequencesException e) {
              System.err.println("Too many sequences from AST");
          } catch (DASTNode.TooLongSequenceException e) {
              System.err.println("Too long sequence from AST");
          }
        }
        return false;
    }


    private void addToJson(String methodName, String body, DSubTree ast, List<Sequence> sequences,  String returnType,
    List<String> formalParam, List<String> classTypes ) {
       JSONOutputWrapper out = new JSONOutputWrapper(methodName, body, ast, sequences, returnType, formalParam, classTypes);
       _js.programs.add(out);
   }


    public String buildJson() throws IOException {
        if (_js.programs.isEmpty())
            return null;

        Gson gson = new GsonBuilder().setPrettyPrinting().serializeNulls().create();

        return gson.toJson(_js);

    }

    public int getLineNumber(ASTNode node) {
        return unit.getLineNumber(node.getStartPosition());
    }

    public String getReturnType(MethodDeclaration m){
      String ret;
      if (m.getReturnType2() != null){
          ret = m.getReturnType2().toString();
      }
      else{
          ret = "None";
      }
      return ret;
    }

    public List<String> getFormalParams(MethodDeclaration m){
      ArrayList<String> parameters = new ArrayList<String>();
      for (Object parameter : m.parameters()) {
          VariableDeclaration variableDeclaration = (VariableDeclaration) parameter;
          String type = variableDeclaration.getStructuralProperty(SingleVariableDeclaration.TYPE_PROPERTY).toString();
          for (int i = 0; i < variableDeclaration.getExtraDimensions(); i++) {
              type += "[]";
          }
          parameters.add(type);
      }
      return parameters;
    }


    private boolean okToPrintAST(List<Sequence> sequences) {
         int n = sequences.size();
         if (n == 0  || (n == 1 && sequences.get(0).getCalls().size() <= 1))
             return false;
         return true;
    }
}
