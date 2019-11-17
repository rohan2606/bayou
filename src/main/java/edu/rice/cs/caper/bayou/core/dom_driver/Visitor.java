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
        String javaDoc;
        // Output
        DSubTree ast;

        // Evidence inputs
        List<Sequence> sequences;

        // New Evidences Types
        String returnType;
        List<String> formalParam;
        List<String> classTypes;

        String className;

        public JSONOutputWrapper(String methodName, String body, DSubTree ast, List<Sequence> sequences,  String returnType, List<String> formalParam,
        List<String> classTypes , String javaDoc, String className) {

            this.file = options.file;
            this.method = methodName;
            this.body = body;

            this.ast = ast;

            this.sequences = sequences;

            this.returnType = returnType;
            this.formalParam = formalParam;
            this.classTypes = classTypes;
            this.javaDoc = javaDoc;

            this.className = className;

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

        // synchronized lists
        for (TypeDeclaration cls : classes) {

            // System.out.println(cls);
            allTypes = new ArrayList<>();
            allTypes.addAll(Arrays.asList(cls.getFields()));

            String className = cls.getName().getIdentifier();

            List<String> classTypes = new ArrayList<>();
            for (FieldDeclaration type : allTypes){
              classTypes.add(type.getType().toString());
            }
            // System.out.println(cls);

            List<DSubTree> asts = new ArrayList<>();
            List<String> javaDocs = new ArrayList<>();
            List<String> methodNames = new ArrayList<>();
            List<String> returnTypes = new ArrayList<>();
            List<String> bodys = new ArrayList<>();
            List<List<String>> formalParams = new ArrayList<>();

//            List<MethodDeclaration> allMethods = new ArrayList<>();
            allMethods = new ArrayList<>();
            allMethods.addAll(Arrays.asList(cls.getMethods()));



            List<MethodDeclaration> constructors = allMethods.stream().filter(m -> m.isConstructor()).collect(Collectors.toList());
            List<MethodDeclaration> publicMethods = allMethods.stream().filter(m -> !m.isConstructor() && Modifier.isPublic(m.getModifiers())).collect(Collectors.toList());


            for (MethodDeclaration m : allMethods){
                  // System.out.println(m.resolveBinding());
                  String javadoc = Utils.getJavadoc(m, options.JAVADOC_TYPE);
                  callStack.push(m);
                  DSubTree ast = new DOMMethodDeclaration(m).handle();
                  callStack.pop();
                  if (ast.isValid()) {
                      asts.add(ast);
                      javaDocs.add(javadoc);
                      methodNames.add(m.getName().getIdentifier() + "@" + getLineNumber(m));
                      returnTypes.add(getReturnType(m));
                      bodys.add(m.toString());
                      formalParams.add(getFormalParams(m));
                  }
            }

            for (int i = 0; i < asts.size(); i++) {
                DSubTree ast = asts.get(i);
                String javaDoc = javaDocs.get(i);
                String methodName = methodNames.get(i);
                String returnType = returnTypes.get(i);
                List<String> formalParam = formalParams.get(i);
                String body = bodys.get(i);

                List<Sequence> sequences = new ArrayList<>();
                sequences.add(new Sequence());
                try {
                    ast.updateSequences(sequences, options.MAX_SEQS, options.MAX_SEQ_LENGTH);
                    List<Sequence> uniqSequences = new ArrayList<>(new HashSet<>(sequences));
                    if (uniqSequences.size() > 0)
                        addToJson(methodName, body, ast, uniqSequences, returnType, formalParam, classTypes, javaDoc, className);
                } catch (DASTNode.TooManySequencesException e) {
                    System.err.println("Too many sequences from AST");
                } catch (DASTNode.TooLongSequenceException e) {
                    System.err.println("Too long sequence from AST");
                }
            }
        }

        return false;
    }

    private void addToJson(String methodName, String body, DSubTree ast, List<Sequence> sequences,  String returnType,
    List<String> formalParam, List<String> classTypes , String javaDoc, String className) {
       JSONOutputWrapper out = new JSONOutputWrapper(methodName, body, ast, sequences, returnType, formalParam, classTypes, javaDoc, className);
       _js.programs.add(out);
   }

    public String buildJson() throws IOException {
        if (_js.programs.isEmpty())
            return null;

        Gson gson = new GsonBuilder().setPrettyPrinting().serializeNulls().create();

        return gson.toJson(_js);

    }

    public String getReturnType(MethodDeclaration m){
      String ret;
      if (m.getReturnType2() != null){
          ret = m.getReturnType2().toString();
      }
      else{
          ret = "__Constructor__";
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

    public int getLineNumber(ASTNode node) {
        return unit.getLineNumber(node.getStartPosition());
    }
}
